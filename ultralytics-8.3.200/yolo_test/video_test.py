#!/usr/bin/env python3
"""
YOLO模型视频测试脚本
支持 .pt 和 .onnx 模型对 .mp4 视频进行推理测试
支持 GPU/CPU 设备选择
"""
import argparse
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO


def get_device(device_arg):
    """获取推理设备"""
    if device_arg:
        return device_arg
    # 自动检测
    if torch.cuda.is_available():
        return 'cuda:0'
    return 'cpu'


def test_video(model_path, video_path, output_path=None, conf=0.25, imgsz=1280, show=True, device=None):
    """
    使用YOLO模型测试视频文件

    Args:
        model_path: .pt 或 .onnx 模型文件路径
        video_path: .mp4视频文件路径
        output_path: 输出视频路径（可选，不指定则不保存）
        conf: 置信度阈值
        imgsz: 推理图像尺寸
        show: 是否实时显示结果
        device: 推理设备 (cuda:0 / cpu)
    """
    # 确定设备
    device = get_device(device)

    # 显示设备信息
    print("=" * 50)
    print("设备信息:")
    if 'cuda' in device:
        print(f"  [GPU] {torch.cuda.get_device_name(0)}")
        print(f"  [显存] {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print(f"  [CPU] 使用CPU推理")
    print(f"  [设备] {device}")
    print("=" * 50)

    # 加载模型
    model_path = Path(model_path)
    print(f"\n加载模型: {model_path}")
    print(f"模型格式: {model_path.suffix}")
    model = YOLO(str(model_path))

    # 检查视频文件
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    print(f"测试视频: {video_path}")
    print(f"置信度阈值: {conf}")
    print(f"推理尺寸: {imgsz}")

    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"视频信息: {width}x{height}, {fps}fps, {total_frames}帧")

    # 设置输出视频
    writer = None
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"输出视频: {output_path}")

    print("\n开始推理... (按 'q' 退出)")
    print("-" * 40)

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 推理（指定设备）
            results = model.predict(
                frame,
                conf=conf,
                imgsz=imgsz,
                device=device,
                verbose=False
            )

            # 绘制结果
            annotated_frame = results[0].plot()

            # 显示检测数量
            num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
            progress = f"帧 {frame_count}/{total_frames} | 检测到 {num_detections} 个目标"
            print(f"\r{progress}", end="", flush=True)

            # 保存到输出视频
            if writer:
                writer.write(annotated_frame)

            # 实时显示
            if show:
                cv2.imshow('YOLO Video Test', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n\n用户中断")
                    break

    finally:
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    print(f"\n\n完成! 共处理 {frame_count} 帧")
    if output_path:
        print(f"结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='YOLO模型视频测试 (支持 .pt 和 .onnx)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用 .pt 模型 (GPU)
  python video_test.py -m model.pt -v video.mp4

  # 使用 .onnx 模型
  python video_test.py -m model.onnx -v video.mp4

  # 强制使用 CPU
  python video_test.py -m model.pt -v video.mp4 --device cpu

  # 保存结果视频
  python video_test.py -m model.pt -v video.mp4 -o result.mp4
        """
    )
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='.pt 或 .onnx 模型文件路径')
    parser.add_argument('--video', '-v', type=str, required=True,
                        help='.mp4视频文件路径')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出视频路径（可选）')
    parser.add_argument('--conf', '-c', type=float, default=0.25,
                        help='置信度阈值 (默认: 0.25)')
    parser.add_argument('--imgsz', '-s', type=int, default=1280,
                        help='推理图像尺寸 (默认: 1280)')
    parser.add_argument('--device', '-d', type=str, default=None,
                        help='推理设备: cuda:0 / cpu (默认: 自动检测GPU)')
    parser.add_argument('--no-show', action='store_true',
                        help='不显示实时画面')

    args = parser.parse_args()

    test_video(
        model_path=args.model,
        video_path=args.video,
        output_path=args.output,
        conf=args.conf,
        imgsz=args.imgsz,
        show=not args.no_show,
        device=args.device
    )


if __name__ == '__main__':
    main()
