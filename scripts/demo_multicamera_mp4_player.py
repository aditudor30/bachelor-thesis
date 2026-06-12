"""Redă sincronizat mai multe fișiere MP4 într-un grid multi-cameră."""

import argparse
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np


DEFAULT_SCREENSHOTS_DIR = Path("output/demo_screenshots")
WINDOW_NAME = "Demonstrator multi-cameră 3D MTMC"


def parse_args() -> argparse.Namespace:
    """Definește și parsează argumentele din linia de comandă."""
    parser = argparse.ArgumentParser(
        description="Redă sincronizat fișiere MP4 deja procesate într-un grid multi-cameră."
    )
    parser.add_argument("--videos", nargs="+", type=Path, required=True, help="Fișierele MP4 de intrare.")
    parser.add_argument("--camera-names", nargs="+", default=None, help="Nume opționale pentru camere.")
    parser.add_argument("--output-video", type=Path, default=None, help="Fișierul MP4 combinat de ieșire.")
    parser.add_argument("--fps", type=float, default=10.0, help="FPS pentru redare și export.")
    parser.add_argument("--scale", type=float, default=1.0, help="Factor de scalare pentru fereastra live.")
    parser.add_argument("--panel-width", type=int, default=640, help="Lățimea fiecărui panou.")
    parser.add_argument("--panel-height", type=int, default=360, help="Înălțimea fiecărui panou.")
    parser.add_argument("--start-frame", type=int, default=0, help="Indexul primului frame, 0-based.")
    parser.add_argument("--end-frame", type=int, default=None, help="Indexul ultimului frame, inclusiv.")
    parser.add_argument("--layout", default=None, help="Layout explicit, de exemplu 2x2, 2x3 sau 1x4.")
    frame_group = parser.add_mutually_exclusive_group()
    frame_group.add_argument("--show-frame-id", dest="show_frame_id", action="store_true")
    frame_group.add_argument("--hide-frame-id", dest="show_frame_id", action="store_false")
    parser.set_defaults(show_frame_id=True)
    parser.add_argument("--save-screenshots-dir", type=Path, default=None, help="Director pentru screenshot-uri.")
    parser.add_argument("--no-window", action="store_true", help="Dezactivează fereastra și rulează doar exportul.")
    parser.add_argument("--overwrite", action="store_true", help="Suprascrie video-ul de ieșire dacă există.")
    parser.add_argument(
        "--on-video-end",
        choices=["black", "hold", "stop"],
        default="black",
        help="Comportament când un video se termină înaintea celorlalte.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validează argumentele înainte de deschiderea fișierelor video."""
    if cv2 is None:
        raise RuntimeError(
            "Acest demonstrator necesită OpenCV. Instalează pachetul opencv-python în mediul de vizualizare."
        )
    if not args.videos:
        raise ValueError("Trebuie specificat cel puțin un fișier prin --videos.")
    for video_path in args.videos:
        if not video_path.exists() or not video_path.is_file():
            raise FileNotFoundError("Fișierul video nu există: %s" % video_path)
        if video_path.suffix.lower() != ".mp4":
            raise ValueError("Demonstratorul acceptă doar fișiere .mp4: %s" % video_path)
    if args.camera_names is not None and len(args.camera_names) != len(args.videos):
        raise ValueError("Numărul de valori --camera-names trebuie să fie egal cu numărul de video-uri.")
    if args.fps <= 0.0:
        raise ValueError("--fps trebuie să fie mai mare decât zero.")
    if args.scale <= 0.0:
        raise ValueError("--scale trebuie să fie mai mare decât zero.")
    if args.panel_width <= 0 or args.panel_height <= 0:
        raise ValueError("Dimensiunile panoului trebuie să fie pozitive.")
    if args.start_frame < 0:
        raise ValueError("--start-frame trebuie să fie 0 sau mai mare.")
    if args.end_frame is not None and args.end_frame < args.start_frame:
        raise ValueError("--end-frame trebuie să fie mai mare sau egal cu --start-frame.")
    if args.no_window and args.output_video is None:
        raise ValueError("--no-window necesită și --output-video.")
    if not args.no_window and not display_is_available():
        raise RuntimeError(
            "Nu a fost detectat un display disponibil. Folosește --no-window împreună cu --output-video."
        )
    if args.output_video is not None and args.output_video.exists() and not args.overwrite:
        raise FileExistsError(
            "Fișierul de ieșire există deja: %s. Folosește --overwrite pentru suprascriere."
            % args.output_video
        )
    rows, cols = compute_grid_layout(len(args.videos), args.layout)
    if rows * cols < len(args.videos):
        raise ValueError("Layout-ul %dx%d nu are suficiente celule pentru %d video-uri." % (rows, cols, len(args.videos)))


def display_is_available() -> bool:
    """Verifică minimal dacă sistemul poate deschide o fereastră OpenCV."""
    if os.name == "nt":
        return True
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return True
    return False


def open_video_streams(video_paths: List[Path], camera_names: List[str]) -> List[Dict[str, Any]]:
    """Deschide toate fișierele video și colectează metadatele lor."""
    streams: List[Dict[str, Any]] = []
    try:
        for video_path, camera_name in zip(video_paths, camera_names):
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                capture.release()
                raise IOError("Video-ul nu poate fi deschis: %s" % video_path)
            metadata = get_video_metadata(capture, video_path)
            streams.append(
                {
                    "path": video_path,
                    "camera_name": camera_name,
                    "capture": capture,
                    "metadata": metadata,
                    "last_frame": None,
                    "last_frame_index": None,
                }
            )
    except Exception:
        release_video_streams(streams)
        raise
    return streams


def get_video_metadata(capture: Any, video_path: Path) -> Dict[str, Any]:
    """Citește metadatele disponibile prin OpenCV."""
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    return {
        "path": str(video_path),
        "frame_count": frame_count if frame_count > 0 else None,
        "width": width if width > 0 else None,
        "height": height if height > 0 else None,
        "fps": fps if fps > 0.0 else None,
    }


def compute_grid_layout(num_videos: int, layout: Optional[str] = None) -> Tuple[int, int]:
    """Calculează numărul de rânduri și coloane al gridului."""
    if num_videos <= 0:
        raise ValueError("Numărul de video-uri trebuie să fie pozitiv.")
    if layout is not None:
        text = layout.lower().strip()
        parts = text.split("x")
        if len(parts) != 2:
            raise ValueError("--layout trebuie să aibă forma RxC, de exemplu 2x2.")
        try:
            rows = int(parts[0])
            cols = int(parts[1])
        except ValueError:
            raise ValueError("--layout trebuie să conțină două numere întregi, de exemplu 2x2.")
        if rows <= 0 or cols <= 0:
            raise ValueError("Valorile din --layout trebuie să fie pozitive.")
        return rows, cols
    if num_videos == 1:
        return 1, 1
    if num_videos == 2:
        return 1, 2
    if num_videos <= 4:
        return 2, 2
    if num_videos <= 6:
        return 2, 3
    if num_videos <= 9:
        return 3, 3
    cols = int(math.ceil(math.sqrt(float(num_videos))))
    rows = int(math.ceil(float(num_videos) / float(cols)))
    return rows, cols


def resize_with_padding(frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Redimensionează cadrul fără deformare și completează marginile cu negru."""
    if frame is None or frame.size == 0:
        return np.zeros((target_height, target_width, 3), dtype=np.uint8)
    source_height, source_width = frame.shape[:2]
    scale = min(float(target_width) / float(source_width), float(target_height) / float(source_height))
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized
    return canvas


def make_camera_panel(
    frame: np.ndarray,
    camera_name: str,
    panel_width: int,
    panel_height: int,
    status_text: Optional[str] = None,
) -> np.ndarray:
    """Construiește un panou video etichetat cu numele camerei."""
    panel = resize_with_padding(frame, panel_width, panel_height)
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (panel_width, 38), (0, 0, 0), thickness=-1)
    panel = cv2.addWeighted(overlay, 0.72, panel, 0.28, 0.0)
    cv2.putText(panel, camera_name, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    if status_text:
        cv2.putText(
            panel,
            status_text,
            (12, panel_height - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (80, 200, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.rectangle(panel, (0, 0), (panel_width - 1, panel_height - 1), (100, 100, 100), 1)
    return panel

def make_empty_panel(
    camera_name: str,
    panel_width: int,
    panel_height: int,
    message: str = "FĂRĂ FRAME DISPONIBIL",
) -> np.ndarray:
    """Construiește un panou negru pentru o celulă fără frame disponibil."""
    panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    cv2.putText(panel, camera_name, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    text_size, _baseline = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    x = max(10, (panel_width - text_size[0]) // 2)
    y = panel_height // 2
    cv2.putText(panel, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.rectangle(panel, (0, 0), (panel_width - 1, panel_height - 1), (100, 100, 100), 1)
    return panel


def make_grid_frame(
    frames: List[Optional[np.ndarray]],
    camera_names: List[str],
    statuses: List[str],
    rows: int,
    cols: int,
    panel_width: int,
    panel_height: int,
    frame_id: int,
    show_frame_id: bool,
) -> np.ndarray:
    """Construiește cadrul final al gridului multi-cameră."""
    panels: List[np.ndarray] = []
    total_cells = rows * cols
    for index in range(total_cells):
        if index >= len(frames):
            panels.append(make_empty_panel("", panel_width, panel_height, ""))
            continue
        frame = frames[index]
        status = statuses[index]
        if frame is None:
            message = "VIDEO TERMINAT" if status == "ended" else "FRAME INDISPONIBIL"
            panels.append(make_empty_panel(camera_names[index], panel_width, panel_height, message))
        else:
            status_text = "ULTIMUL FRAME" if status == "held" else None
            panels.append(make_camera_panel(frame, camera_names[index], panel_width, panel_height, status_text))
    grid_rows = []
    for row_index in range(rows):
        start = row_index * cols
        grid_rows.append(np.hstack(panels[start : start + cols]))
    grid = np.vstack(grid_rows)
    if show_frame_id:
        text = "Frame %06d" % frame_id
        text_size, _baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        x = max(8, grid.shape[1] - text_size[0] - 14)
        cv2.rectangle(grid, (x - 8, 4), (grid.shape[1] - 4, 36), (0, 0, 0), thickness=-1)
        cv2.putText(grid, text, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return grid


def save_screenshot(grid_frame: np.ndarray, frame_id: int, screenshots_dir: Optional[Path]) -> Path:
    """Salvează un screenshot al gridului curent."""
    output_dir = screenshots_dir if screenshots_dir is not None else DEFAULT_SCREENSHOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / ("multicamera_frame_%06d.png" % frame_id)
    if not cv2.imwrite(str(output_path), grid_frame):
        raise IOError("Screenshot-ul nu a putut fi salvat: %s" % output_path)
    return output_path


def create_video_writer(output_path: Path, fps: float, frame_size: Tuple[int, int]) -> Any:
    """Creează writer-ul MP4 pentru gridul combinat."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), frame_size)
    if not writer.isOpened():
        writer.release()
        raise IOError("Video-ul de ieșire nu poate fi creat: %s" % output_path)
    return writer


def read_synchronized_frames(
    streams: List[Dict[str, Any]],
    frame_index: int,
    on_video_end: str,
) -> Tuple[List[Optional[np.ndarray]], List[str], bool, bool]:
    """Citește același index de frame din toate video-urile."""
    frames: List[Optional[np.ndarray]] = []
    statuses: List[str] = []
    ended_count = 0
    stop_requested = False
    for stream in streams:
        metadata = stream["metadata"]
        frame_count = metadata.get("frame_count")
        is_past_end = frame_count is not None and frame_index >= int(frame_count)
        if is_past_end:
            ended_count += 1
            if on_video_end == "stop":
                stop_requested = True
            frame = held_last_frame(stream) if on_video_end == "hold" else None
            frames.append(frame)
            statuses.append("held" if frame is not None else "ended")
            continue
        capture = stream["capture"]
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok or frame is None:
            ended_count += 1
            if on_video_end == "stop":
                stop_requested = True
            held = held_last_frame(stream) if on_video_end == "hold" else None
            frames.append(held)
            statuses.append("held" if held is not None else "ended")
            continue
        stream["last_frame"] = frame.copy()
        stream["last_frame_index"] = frame_index
        frames.append(frame)
        statuses.append("ok")
    return frames, statuses, stop_requested, ended_count == len(streams)


def held_last_frame(stream: Dict[str, Any]) -> Optional[np.ndarray]:
    """Returnează ultimul frame disponibil pentru modul hold."""
    cached = stream.get("last_frame")
    metadata = stream.get("metadata", {})
    frame_count = metadata.get("frame_count")
    if frame_count is not None and int(frame_count) > 0:
        last_index = int(frame_count) - 1
        if stream.get("last_frame_index") != last_index:
            capture = stream["capture"]
            capture.set(cv2.CAP_PROP_POS_FRAMES, last_index)
            ok, frame = capture.read()
            if ok and frame is not None:
                stream["last_frame"] = frame.copy()
                stream["last_frame_index"] = last_index
                cached = frame
    return None if cached is None else cached.copy()


def determine_end_frame(
    streams: List[Dict[str, Any]],
    requested_end: Optional[int],
    on_video_end: str,
) -> Optional[int]:
    """Determină ultimul frame pe baza lungimilor video și a politicii de final."""
    if requested_end is not None:
        return requested_end
    counts = [stream["metadata"].get("frame_count") for stream in streams]
    known_counts = [int(value) for value in counts if value is not None and int(value) > 0]
    if not known_counts:
        return None
    if on_video_end == "stop":
        return min(known_counts) - 1
    return max(known_counts) - 1


def run_live_player(
    streams: List[Dict[str, Any]],
    camera_names: List[str],
    args: argparse.Namespace,
    rows: int,
    cols: int,
    end_frame: Optional[int],
) -> None:
    """Rulează redarea interactivă și, opțional, exportul video."""
    writer = None
    frame_size = (cols * args.panel_width, rows * args.panel_height)
    if args.output_video is not None:
        writer = create_video_writer(args.output_video, args.fps, frame_size)
    current_frame = args.start_frame
    playing = True
    speed = 1.0
    grid_frame: Optional[np.ndarray] = None
    rendered_frame: Optional[int] = None
    written_frames = set()
    try:
        while True:
            if end_frame is not None and current_frame > end_frame:
                break
            if rendered_frame != current_frame or grid_frame is None:
                frames, statuses, stop_requested, all_ended = read_synchronized_frames(
                    streams, current_frame, args.on_video_end
                )
                if stop_requested or (all_ended and end_frame is None):
                    break
                grid_frame = make_grid_frame(
                    frames,
                    camera_names,
                    statuses,
                    rows,
                    cols,
                    args.panel_width,
                    args.panel_height,
                    current_frame,
                    args.show_frame_id,
                )
                rendered_frame = current_frame
                if writer is not None and current_frame not in written_frames:
                    writer.write(grid_frame)
                    written_frames.add(current_frame)
            display_frame = grid_frame
            if args.scale != 1.0:
                display_width = max(1, int(round(grid_frame.shape[1] * args.scale)))
                display_height = max(1, int(round(grid_frame.shape[0] * args.scale)))
                display_frame = cv2.resize(grid_frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
            cv2.imshow(WINDOW_NAME, display_frame)
            delay_ms = max(1, int(round(1000.0 / (args.fps * speed)))) if playing else 30
            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                playing = not playing
            elif key in (ord("d"), ord("D")):
                playing = False
                current_frame += 1
            elif key in (ord("a"), ord("A")):
                playing = False
                current_frame = max(args.start_frame, current_frame - 1)
            elif key in (ord("f"), ord("F")):
                speed = min(8.0, speed * 1.25)
                print("[INFO] Viteză redare: %.2fx" % speed)
            elif key in (ord("s"), ord("S")):
                speed = max(0.125, speed / 1.25)
                print("[INFO] Viteză redare: %.2fx" % speed)
            elif key in (ord("r"), ord("R")):
                playing = False
                current_frame = args.start_frame
            elif key in (ord("p"), ord("P")):
                screenshot = save_screenshot(grid_frame, current_frame, args.save_screenshots_dir)
                print("[INFO] Screenshot salvat: %s" % screenshot)
            elif playing:
                current_frame += 1
    finally:
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def run_export_only(
    streams: List[Dict[str, Any]],
    camera_names: List[str],
    args: argparse.Namespace,
    rows: int,
    cols: int,
    end_frame: Optional[int],
) -> None:
    """Exportă determinist gridul fără a deschide o fereastră."""
    if args.output_video is None:
        raise ValueError("Modul export-only necesită --output-video.")
    frame_size = (cols * args.panel_width, rows * args.panel_height)
    writer = create_video_writer(args.output_video, args.fps, frame_size)
    current_frame = args.start_frame
    exported = 0
    last_report_time = time.time()
    try:
        while True:
            if end_frame is not None and current_frame > end_frame:
                break
            frames, statuses, stop_requested, all_ended = read_synchronized_frames(
                streams, current_frame, args.on_video_end
            )
            if stop_requested or (all_ended and end_frame is None):
                break
            grid_frame = make_grid_frame(
                frames,
                camera_names,
                statuses,
                rows,
                cols,
                args.panel_width,
                args.panel_height,
                current_frame,
                args.show_frame_id,
            )
            writer.write(grid_frame)
            exported += 1
            now = time.time()
            if exported == 1 or exported % 100 == 0 or now - last_report_time >= 10.0:
                if end_frame is None:
                    print("[INFO] Export: %d frame-uri" % exported)
                else:
                    total = max(0, end_frame - args.start_frame + 1)
                    print("[INFO] Export: %d/%d frame-uri" % (exported, total))
                last_report_time = now
            current_frame += 1
    finally:
        writer.release()
    print("[INFO] Export finalizat: %s" % args.output_video)
    print("[INFO] Frame-uri exportate: %d" % exported)


def release_video_streams(streams: List[Dict[str, Any]]) -> None:
    """Eliberează toate obiectele VideoCapture."""
    for stream in streams:
        capture = stream.get("capture")
        if capture is not None:
            capture.release()


def print_startup_info(
    streams: List[Dict[str, Any]],
    rows: int,
    cols: int,
    args: argparse.Namespace,
    end_frame: Optional[int],
) -> None:
    """Afișează configurația demonstratorului în terminal."""
    print("[INFO] Demonstrator multi-cameră")
    print("[INFO] Video-uri încărcate:")
    for stream in streams:
        print("       %s -> %s" % (stream["camera_name"], stream["path"]))
    print("[INFO] Layout: %dx%d" % (rows, cols))
    print("[INFO] Dimensiune panou: %dx%d" % (args.panel_width, args.panel_height))
    print("[INFO] Interval frame-uri: %d -> %s" % (args.start_frame, "automat" if end_frame is None else end_frame))
    print("[INFO] FPS: %s" % args.fps)
    print("[INFO] Comportament final video: %s" % args.on_video_end)
    print("[INFO] Output video: %s" % (args.output_video if args.output_video is not None else "dezactivat"))
    if not args.no_window:
        print("[INFO] Controale: Space=pauză/play, A=înapoi, D=înainte, F=mai rapid, S=mai lent, R=restart, P=screenshot, Q/ESC=ieșire")


def main() -> None:
    """Punctul principal de intrare al demonstratorului."""
    args = parse_args()
    validate_args(args)
    camera_names = args.camera_names if args.camera_names is not None else [path.stem for path in args.videos]
    rows, cols = compute_grid_layout(len(args.videos), args.layout)
    streams = open_video_streams(args.videos, camera_names)
    try:
        end_frame = determine_end_frame(streams, args.end_frame, args.on_video_end)
        print_startup_info(streams, rows, cols, args, end_frame)
        if args.no_window:
            run_export_only(streams, camera_names, args, rows, cols, end_frame)
        else:
            run_live_player(streams, camera_names, args, rows, cols, end_frame)
    finally:
        release_video_streams(streams)


if __name__ == "__main__":
    main()
