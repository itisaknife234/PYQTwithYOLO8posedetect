import sys
import cv2
import numpy as np
import imageio
from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout,
    QLineEdit, QHBoxLayout, QRadioButton, QButtonGroup, QSlider
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QImage
from ultralytics import YOLO


class PoseDetectionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("YOLO Pose Detection")
        self.setGeometry(100, 100, 900, 750)

        self.file_input = QLineEdit(self)
        self.file_input.setPlaceholderText("파일 경로를 입력하거나 '파일 선택' 버튼을 클릭하세요.")

        self.btn_select = QPushButton("파일 선택", self)
        self.btn_select.clicked.connect(self.select_file)

        self.radio_image = QRadioButton("이미지 감지", self)
        self.radio_gif = QRadioButton("GIF 감지", self)
        self.radio_video = QRadioButton("영상 감지", self)
        self.radio_image.setChecked(True)

        self.radio_group = QButtonGroup(self)
        self.radio_group.addButton(self.radio_image)
        self.radio_group.addButton(self.radio_gif)
        self.radio_group.addButton(self.radio_video)

        self.btn_detect = QPushButton("Pose Detection 실행", self)
        self.btn_detect.setEnabled(False)
        self.btn_detect.clicked.connect(self.run_pose_detection)

        self.btn_pause = QPushButton("일시정지 / 재개", self)
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self.toggle_pause)

        self.btn_screenshot = QPushButton("스크린샷 저장", self)
        self.btn_screenshot.setEnabled(False)
        self.btn_screenshot.clicked.connect(self.save_screenshot)

        self.slider_timeline = QSlider(Qt.Orientation.Horizontal, self)
        self.slider_timeline.setMinimum(0)
        self.slider_timeline.setEnabled(False)
        self.slider_timeline.valueChanged.connect(self.seek_video)

        self.time_label = QLabel("00:00 / 00:00", self)
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.display_label = QLabel(self)
        self.display_label.setFixedSize(800, 500)
        self.display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.display_label.setStyleSheet("border: 2px solid black; background-color: lightgray;")

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.file_input)
        input_layout.addWidget(self.btn_select)

        radio_layout = QHBoxLayout()
        radio_layout.addWidget(self.radio_image)
        radio_layout.addWidget(self.radio_gif)
        radio_layout.addWidget(self.radio_video)

        video_control_layout = QHBoxLayout()
        video_control_layout.addWidget(self.btn_pause)
        video_control_layout.addWidget(self.btn_screenshot)

        main_layout = QVBoxLayout()
        main_layout.addLayout(input_layout)
        main_layout.addLayout(radio_layout)
        main_layout.addWidget(self.btn_detect)
        main_layout.addLayout(video_control_layout)
        main_layout.addWidget(self.display_label)
        main_layout.addWidget(self.slider_timeline)
        main_layout.addWidget(self.time_label)

        self.setLayout(main_layout)

        self.file_path = None
        self.model_path = "yolov8n-pose.pt"
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_video_frame)
        self.pause = False
        self.total_frames = 0
        self.fps = 0
        self.displayed_frame = None

    def select_file(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)

        if self.radio_image.isChecked():
            file_types = "이미지 파일 (*.png *.jpg *.jpeg)"
        elif self.radio_gif.isChecked():
            file_types = "GIF 파일 (*.gif)"
        elif self.radio_video.isChecked():
            file_types = "비디오 파일 (*.mp4 *.avi *.mov)"
        else:
            file_types = "모든 파일 (*.*)"

        file_path, _ = file_dialog.getOpenFileName(self, "파일 선택", "", file_types)

        if file_path:
            self.file_path = file_path
            self.file_input.setText(file_path)
            self.btn_detect.setEnabled(True)

    def run_pose_detection(self):
        self.file_path = self.file_input.text().strip()

        if not self.file_path:
            print("파일을 입력하거나 선택하세요.")
            return

        model = YOLO(self.model_path)

        if self.radio_image.isChecked():
            self.detect_image(model)
        elif self.radio_gif.isChecked():
            self.detect_gif(model)
        elif self.radio_video.isChecked():
            self.detect_video(model)
        else:
            print("지원되지 않는 파일 형식입니다.")

    def detect_video(self, model):
        self.cap = cv2.VideoCapture(self.file_path)
        if not self.cap.isOpened():
            print("영상을 불러올 수 없습니다.")
            return
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider_timeline.setMaximum(self.total_frames)
        self.slider_timeline.setEnabled(True)
        self.btn_pause.setEnabled(True)
        self.btn_screenshot.setEnabled(True)
        self.timer.start(30)

    def update_video_frame(self):
        if not self.pause:
            ret, frame = self.cap.read()
            if not ret:
                self.timer.stop()
                self.cap.release()
                return
            
            model = YOLO(self.model_path)
            results = model(frame)
            self.displayed_frame = results[0].plot()

            frame_rgb = cv2.cvtColor(self.displayed_frame, cv2.COLOR_BGR2RGB)

            label_width = self.display_label.width()
            label_height = self.display_label.height()
            frame_resized = cv2.resize(frame_rgb, (label_width, label_height), interpolation=cv2.INTER_AREA)

            h, w, ch = frame_resized.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_resized.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)

            self.display_label.setPixmap(pixmap.scaled(label_width, label_height, Qt.AspectRatioMode.KeepAspectRatio))
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.slider_timeline.setValue(current_frame)
            self.update_time_label(current_frame)

    def update_time_label(self, current_frame):
        current_time = int(current_frame / self.fps)
        total_time = int(self.total_frames / self.fps)
        self.time_label.setText(f"{current_time // 60}:{current_time % 60:02d} / {total_time // 60}:{total_time % 60:02d}")

    def toggle_pause(self):
        self.pause = not self.pause

    def save_screenshot(self):
        if self.displayed_frame is not None:
            cv2.imwrite("screenshot.jpg", self.displayed_frame)
            print("스크린샷이 저장되었습니다: screenshot.jpg")

    def seek_video(self, frame_number):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PoseDetectionApp()
    window.show()
    sys.exit(app.exec())
