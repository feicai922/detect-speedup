import argparse
import math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    import rasterio
    from rasterio.windows import Window
    from rasterio.transform import from_bounds
except ImportError:
    print("请安装 rasterio: pip install rasterio")
    exit(1)

ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".gif"}
TILE_SIZE = 1024
WHITE_THRESHOLD = 240
ALPHA_THRESHOLD = 240


def get_tile_geoinfo(src, x0, y0, x1, y1):
    """
    获取瓦片的地理坐标信息
    返回: (左上角经度/东向, 左上角纬度/北向, 右下角经度/东向, 右下角纬度/北向)
    """
    transform = src.transform
    
    # 计算四个角的地理坐标
    left, top = transform * (x0, y0)
    right, bottom = transform * (x1, y1)
    
    return {
        'xmin': left,
        'ymin': bottom,
        'xmax': right,
        'ymax': top
    }


def has_content(tile_data: np.ndarray, threshold: int = WHITE_THRESHOLD) -> bool:
    """检查瓦片是否有内容（非纯白或非透明）"""
    if tile_data is None or tile_data.size == 0:
        return False
    
    # 采样检测以提高速度
    if tile_data.ndim == 2:
        h, w = tile_data.shape
        step = max(1, min(h, w) // 64)
        sampled = tile_data[::step, ::step]
        return np.any(sampled < threshold)
    elif tile_data.ndim == 3:
        h, w, c = tile_data.shape
        step = max(1, min(h, w) // 64)
        sampled = tile_data[::step, ::step]
        
        if c == 4:
            alpha = sampled[:, :, 3]
            if np.any(alpha < ALPHA_THRESHOLD):
                return True
            rgb = sampled[:, :, :3]
            return np.any(np.any(rgb < threshold, axis=2))
        elif c >= 3:
            return np.any(np.any(sampled[:, :, :3] < threshold, axis=2))
        else:
            return np.any(sampled < threshold)
    
    return False


def process_single_tile(args):
    """处理单个瓦片（用于多进程）"""
    image_path, output_dir, tile_size, row, col, x0, y0, x1, y1 = args
    
    try:
        with rasterio.open(image_path) as src:
            # 确保坐标是整数
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            width = x1 - x0
            height = y1 - y0
            
            # 获取地理信息
            geo_info = get_tile_geoinfo(src, x0, y0, x1, y1)
            
            # 读取窗口
            window = Window(col_off=x0, row_off=y0, width=width, height=height)
            tile_data = src.read(window=window)
            
            # 转置为 (height, width, channels)
            if tile_data.ndim == 3:
                tile_data = np.transpose(tile_data, (1, 2, 0))
            
            # 检查是否有内容
            if not has_content(tile_data):
                return None
            
            # 归一化到 0-255
            if tile_data.dtype != np.uint8:
                if tile_data.max() > 0:
                    tile_data = (tile_data / tile_data.max() * 255).astype(np.uint8)
                else:
                    tile_data = tile_data.astype(np.uint8)
            
            # 准备profile用于保存GeoTIFF
            profile = src.profile.copy()
            profile.update({
                'dtype': 'uint8',
                'compress': 'lzw'
            })
            
            # 调整维度顺序为 (bands, height, width)
            if tile_data.ndim == 3:
                img_array = np.transpose(tile_data, (2, 0, 1))
                profile.update({'count': tile_data.shape[2]})
            else:
                img_array = tile_data[np.newaxis, :, :]
                profile.update({'count': 1})
            
            # 更新transform
            transform = from_bounds(
                geo_info['xmin'], geo_info['ymin'],
                geo_info['xmax'], geo_info['ymax'],
                img_array.shape[2], img_array.shape[1]
            )
            
            profile.update({
                'transform': transform,
                'width': img_array.shape[2],
                'height': img_array.shape[1],
                'driver': 'GTiff'
            })
            
            # 文件名包含地理坐标（使用小数点后6位，约0.1米精度）
            xmin_str = f"{geo_info['xmin']:.6f}".replace('.', '_')
            ymin_str = f"{geo_info['ymin']:.6f}".replace('.', '_')
            xmax_str = f"{geo_info['xmax']:.6f}".replace('.', '_')
            ymax_str = f"{geo_info['ymax']:.6f}".replace('.', '_')
            
            tile_name = f"{Path(image_path).stem}_{row:03d}_{col:03d}_{xmin_str}_{ymin_str}_{xmax_str}_{ymax_str}.tif"
            tile_path = Path(output_dir) / tile_name
            
            # 保存为GeoTIFF（保留地理信息）
            with rasterio.open(tile_path, 'w', **profile) as dst:
                dst.write(img_array)
            
            return tile_path
        
    except Exception as e:
        print(f"处理瓦片 (row={row}, col={col}) 失败: {e}")
        return None


def split_image_parallel(image_path: Path, output_dir: Path, tile_size: int, num_workers: int = 8) -> int:
    """使用多进程分块读取大图"""
    try:
        with rasterio.open(image_path) as src:
            width = src.width
            height = src.height
            print(f"图片尺寸: {width} x {height}")
            print(f"CRS: {src.crs}")
            
            cols = math.ceil(width / tile_size)
            rows = math.ceil(height / tile_size)
            total_tiles = cols * rows
            
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 准备所有任务
            tasks = []
            for row in range(rows):
                y0 = row * tile_size
                y1 = min(y0 + tile_size, height)
                for col in range(cols):
                    x0 = col * tile_size
                    x1 = min(x0 + tile_size, width)
                    tasks.append((
                        str(image_path),
                        str(output_dir),
                        tile_size,
                        row, col,
                        x0, y0, x1, y1
                    ))
            
            print(f"生成 {len(tasks)} 个瓦片任务，使用 {num_workers} 个进程...")
            
            # 多进程处理
            saved_tiles = 0
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(process_single_tile, task): task for task in tasks}
                
                with tqdm(total=len(tasks), desc=f"切分 {image_path.name}") as pbar:
                    for future in as_completed(futures):
                        result = future.result()
                        if result is not None:
                            saved_tiles += 1
                        pbar.update(1)
            
            return saved_tiles
            
    except Exception as e:
        print(f"打开图片失败: {e}")
        import traceback
        traceback.print_exc()
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将大图切成指定大小的瓦片，并跳过纯白区域（保留地理坐标）")
    parser.add_argument("--input", "-i", type=str, default=str(ROOT), help="输入目录")
    parser.add_argument("--output", "-o", type=str, default=str(ROOT / "tiles"), help="输出目录")
    parser.add_argument("--tile-size", "-s", type=int, default=TILE_SIZE, help="瓦片大小，默认 20000")
    parser.add_argument("--workers", "-w", type=int, default=8, help="并行进程数，默认 8")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"输入目录不存在: {input_dir}")
        return
    
    # 查找所有tif文件
    image_paths = []
    for ext in ['.tif', '.tiff', '.TIF', '.TIFF']:
        image_paths.extend(input_dir.glob(f"*{ext}"))
    
    image_paths = sorted(set(image_paths))
    
    if not image_paths:
        print(f"未找到可处理的TIFF文件，输入目录: {input_dir}")
        return
    
    print(f"找到 {len(image_paths)} 张大图，开始处理...\n")
    
    for image_path in image_paths:
        try:
            output_dir = output_root / image_path.stem
            saved = split_image_parallel(image_path, output_dir, args.tile_size, args.workers)
            print(f"\n✅ 完成: {image_path.name} -> {saved} 张有效切片，输出目录: {output_dir}\n")
        except Exception as e:
            print(f"❌ 处理 {image_path.name} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("全部处理完成。")


if __name__ == "__main__":
    main()