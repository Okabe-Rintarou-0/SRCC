import os
import shutil
import sys
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import QRectF, QDir, Qt
from PyQt5.QtGui import QImage, QPainter, QPixmap
from PyQt5.QtWidgets import (QApplication, QGraphicsScene, QGraphicsView,
                             QFileDialog, QMainWindow)

from predict import predict
from utils.plots import Annotator, Colors
from viewer_ui import Ui_MainWindow


class Viewer(QMainWindow):
    def __init__(self):
        super().__init__()
        # 使用Ui_MainWindow类来设置主窗口的UI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("SRCC Viewer")
        self.ui.detect_btn.clicked.connect(self.detect)
        self.ui.choose_btn.clicked.connect(self.choose_img)
        self.org_path = "./org"
        self.cur_image: Optional[str] = None
        self.result_path = "./out"
        self.ui.result_view.setScene(QGraphicsScene())
        self.ui.org_path_editor.setText(Viewer.absolute_path(self.org_path))
        self.ui.result_path_editor.setText(Viewer.absolute_path(self.result_path))
        self.ui.org_path_browser_btn.clicked.connect(self.choose_org_dir)
        self.ui.result_path_browser_btn.clicked.connect(self.choose_result_dir)
        self.colors = Colors()
        Viewer.check_create_dir(self.org_path)
        Viewer.check_create_dir(self.result_path)

    @staticmethod
    def absolute_path(relative_path: str) -> str:
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        # 将相对路径转换为绝对路径
        return os.path.abspath(os.path.join(cur_dir, relative_path))

    @staticmethod
    def check_create_dir(dir_path: str):
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

    @staticmethod
    def clear_dir(path: str):
        for root, dirs, files in os.walk(path):
            for file in files:
                os.remove(os.path.join(root, file))

    @staticmethod
    def get_available_file_path(path: str):
        ext = os.path.splitext(path)[1]
        i = 0
        while True:
            i += 1
            new_path = path.replace(ext, f'_{i}{ext}')
            if not os.path.exists(new_path):
                return new_path

    def draw_boxes(self, img, boxes):
        annotator = Annotator(img, line_width=3, example='rcc')
        for box in boxes:
            _, prob, xyxy = box
            annotator.box_label(xyxy, f'rcc {prob / 100:.2f}', color=self.colors(0, True))
        return annotator.result()

    def analyze_boxes(self, boxes):
        num = len(boxes)
        self.ui.num_rcc_label.setText(str(num))
        avg_w = 0
        avg_h = 0
        for box in boxes:
            _, _, xyxy = box
            x_min, y_min, x_max, y_max = xyxy
            w = x_max - x_min
            h = y_max - y_min
            avg_w += w / num
            avg_h += h / num
        self.ui.avg_width_label.setText(f"{avg_w:.2f}")
        self.ui.avg_height_label.setText(f"{avg_h:.2f}")

    def detect(self):
        if not self.cur_image:
            return
        # 临时禁用滚动条
        self.ui.org_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ui.org_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # 截图
        screenshot = self.ui.org_view.grab()

        # 恢复滚动条
        self.ui.org_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.ui.org_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

        image_path = os.path.join(self.org_path, self.cur_image)
        if os.path.exists(image_path):
            image_path = Viewer.get_available_file_path(image_path)

        # 保存截图到图片文件
        screenshot.save(image_path)
        img = cv2.imread(image_path)
        boxes = predict(img)
        self.analyze_boxes(boxes)

        img = self.draw_boxes(img, boxes)
        result_img_path = os.path.join(self.result_path, self.cur_image)
        if os.path.exists(result_img_path):
            result_img_path = Viewer.get_available_file_path(result_img_path)
        cv2.imwrite(result_img_path, img)

        q_img = self.CVMat_to_QImage(img)

        result_scene = self.ui.result_view.scene()
        result_scene.clear()
        pixmap = QPixmap.fromImage(q_img)
        result_scene.addPixmap(pixmap)

    @staticmethod
    def QImage_to_CVMat(image):
        '''  Converts a QImage into an opencv MAT format  '''

        size = image.size()
        s = image.bits().asstring(size.width() * size.height() * image.depth() // 8)  # format 0xffRRGGBB

        mat = np.frombuffer(s, dtype=np.uint8).reshape((size.height(), size.width(), image.depth() // 8))
        mat_rgb = cv2.cvtColor(mat, cv2.COLOR_RGBA2RGB)
        return mat_rgb

    @staticmethod
    def CVMat_to_QImage(img):
        # 转换为RGB格式
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 转换为QImage
        height, width, channel = img.shape
        bytesPerLine = 3 * width
        qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)

        return qImg

    def on_mouse_wheel(self, event):
        # 检查滚动条的位置
        is_scrollbar_visible = self.ui.org_view.horizontalScrollBar().isVisible() or \
                               self.ui.org_view.verticalScrollBar().isVisible()

        # 如果滚动条可见，则禁用滚动条
        if is_scrollbar_visible:
            self.ui.org_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.ui.org_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # 执行缩放操作
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            if event.angleDelta().y() > 0:
                self.ui.org_view.scale(1.1, 1.1)
            else:
                self.ui.org_view.scale(0.9, 0.9)

        # 恢复滚动条的状态
        if is_scrollbar_visible:
            self.ui.org_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
            self.ui.org_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

        event.accept()

    def show_region(self, x, y, width, height):
        # 设置视口的位置和大小
        self.ui.org_view.setSceneRect(QRectF(x, y, width, height))

        # 在视图中居中显示指定的区域
        self.ui.org_view.centerOn(x + width / 2, y + height / 2)

    def get_dir_path(self, start_dir: str) -> str:
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        dialog.setDirectory(start_dir)
        # 设置只能选择目录
        dialog.setOption(QFileDialog.ShowDirsOnly)
        if dialog.exec_() == QFileDialog.Accepted:
            directory = dialog.selectedFiles()[0]
            print("选择的目录为:", directory)
            return directory

    def get_img_path(self, start_dir: str) -> str:
        dialog = QFileDialog()
        file_filter = "Images (*.png *.xpm *.jpg *.bmp *.jpeg);;All Files (*)"
        dialog.setDirectory(start_dir)
        img_path, _ = dialog.getOpenFileName(self, "选择文件", "", file_filter)
        return img_path

    def choose_result_dir(self):
        dir = self.get_dir_path(self.result_path)
        if dir:
            self.result_path = dir
            self.ui.result_path_editor.setText(dir)

    def choose_org_dir(self):
        dir = self.get_dir_path(self.org_path)
        if dir:
            self.org_path = dir
            self.ui.org_path_editor.setText(dir)

    def choose_img(self):
        current_dir = QDir.currentPath()
        img_path = self.get_img_path(current_dir)

        if img_path:
            print("选中图片：", img_path)
            with open(img_path, "rb") as f:
                image_data = f.read()
                image = QImage.fromData(image_data)
                if image.isNull():
                    print("读取失败")
                    return
                self.cur_image = os.path.basename(img_path)
                self.show_img(image)

    def show_img(self, image: QImage):
        scene = QGraphicsScene()
        pixmap = QPixmap.fromImage(image)
        scene.addPixmap(pixmap)
        self.ui.org_view.setScene(scene)
        self.ui.org_view.setRenderHint(QPainter.Antialiasing)
        self.ui.org_view.setFixedSize(640, 640)

        # 在视图中居中显示图像
        self.ui.org_view.centerOn(pixmap.width() / 2, pixmap.height() / 2)

        # 添加滚动条
        self.ui.org_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.ui.org_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

        # 实现鼠标滚轮缩放
        self.ui.org_view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.ui.org_view.viewport().installEventFilter(self)
        self.ui.org_view.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.ui.org_view.wheelEvent = self.on_mouse_wheel


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = Viewer()
    viewer.show()
    app.exec_()
    sys.exit()
