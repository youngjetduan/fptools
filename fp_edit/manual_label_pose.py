"""
This file (manual_label_pose.py) is designed for:
    manual labeling tool for fingerprint pose
Copyright (c) 2021, Yongjie Duan. All rights reserved.
"""
import os
import sys

# os.chdir(sys.path[0])
import os.path as osp
import numpy as np
from glob import glob
from PyQt5.Qt import QWidget
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QSplitter,
    QLineEdit,
    QLabel,
    QFrame,
    QMessageBox,
)
from PyQt5.QtGui import QIntValidator, QImage, QPixmap, QPainter, QPen
from PyQt5 import QtCore
from pyqtgraph.widgets.RawImageWidget import RawImageWidget


class PainterLabel(QLabel):
    ready = QtCore.pyqtSignal(list)

    def __init__(self, Parent=None):
        super().__init__(Parent)
        self.x1, self.y1 = 0, 0
        self.x2, self.y2 = 0, 0

        self.__pose_pts = []
        self.__is_drawing = False

    def mousePressEvent(self, event):
        self.__pose_pts = [(event.x(), event.y())]
        self.__is_drawing = True

    def mouseMoveEvent(self, event):
        if self.__is_drawing:
            self.x1, self.y1 = self.__pose_pts[0]
            self.x2, self.y2 = event.x(), event.y()
            self.update()

    def mouseReleaseEvent(self, event):
        self.__pose_pts.append((event.x(), event.y()))
        self.__is_drawing = False
        self.draw_pose(self.__pose_pts[0], self.__pose_pts[1])
        self.ready.emit(self.__pose_pts)

    def draw_pose(self, pt1, pt2):
        self.x1, self.y1 = pt1
        self.x2, self.y2 = pt2
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)

        if self.x1 != 0 or self.y1 != 0:
            painter = QPainter(self)

            painter.setPen(QPen(QtCore.Qt.red, 10))
            painter.drawPoint(self.x1, self.y1)

            painter.setPen(QPen(QtCore.Qt.red, 3, QtCore.Qt.SolidLine))
            painter.drawLine(self.x1, self.y1, self.x2, self.y2)


class MainWidget(QWidget):
    def __init__(self, prefix, name_filter="*", ext="png", Parent=None) -> None:
        """Constructor"""
        super().__init__(Parent)

        self.__prefix = prefix
        self.__name_filter = name_filter
        self.__ext = ext
        self.__pose_prefix = osp.join(prefix + "_feature", "pose_m")

        self.__name_lst = glob(osp.join(prefix, f"**/{name_filter}.{ext}"), recursive=True)
        self.__name_lst = [x.replace(f"{prefix}/", "").replace(f".{ext}", "") for x in self.__name_lst]
        self.__name_lst.sort()

        if len(self.__name_lst) == 0:
            QMessageBox.critical(self, "Error", "No image file with specified format is found")
            return

        self.__img_panel_width = 512
        self.__img_panel_height = 512

        self.__InitView()
        self.__SetupConnection()

        self.__scale_ratio = 1.0
        self.__cur_pose = np.zeros(3)
        self.__index = -1
        self.__is_modifed = False
        self.on_btn_Next_Clicked()

    def __InitView(self):
        """初始化界面"""
        # self.setFixedSize(640, 480)
        self.setWindowTitle("Pose Annotation")

        top_layout = QVBoxLayout(self)
        top_layout.setSpacing(10)

        sub_layout = QHBoxLayout()
        sub_layout.setContentsMargins(5, 5, 5, 5)
        self.__label_prefix = QLabel("Image Filter")
        self.__label_prefix.setFixedHeight(20)
        sub_layout.addWidget(self.__label_prefix)
        self.__edit_prefix = QLineEdit(osp.join(self.__prefix, f"**/{self.__name_filter}.{self.__ext}"))
        self.__edit_prefix.setFixedHeight(20)
        self.__edit_prefix.setReadOnly(True)
        sub_layout.addWidget(self.__edit_prefix)
        top_layout.addLayout(sub_layout)

        # 新建一个水平布局作为本窗体的主布局
        main_layout = QHBoxLayout()
        # 设置主布局内边距以及控件间距为10px
        main_layout.setSpacing(10)

        # 新建垂直子布局用于放置画板和提示栏
        sub_layout = QVBoxLayout()
        # 设置此子布局和内部控件的间距为5px
        sub_layout.setContentsMargins(5, 5, 5, 5)
        # 在主界面左侧放置画板
        self.__img_box = PainterLabel()
        self.__img_box.setGeometry(20, 20, self.__img_panel_width, self.__img_panel_height)
        self.__img_box.setAlignment(QtCore.Qt.AlignTop)
        sub_layout.addWidget(self.__img_box)
        self.__label_info = QLabel()
        self.__label_info.setFixedWidth(self.__img_panel_width - 20)
        self.__label_info.setFixedHeight(20)
        self.__label_info.setFrameShape(QFrame.Box)
        self.__label_info.setFrameShadow(QFrame.Shadow.Raised)
        sub_layout.addWidget(self.__label_info)
        main_layout.addLayout(sub_layout)

        # 新建垂直子布局用于放置按键
        sub_layout = QVBoxLayout()
        # 设置此子布局和内部控件的间距为5px
        sub_layout.setContentsMargins(5, 5, 5, 5)
        # image name
        ssub_layout = QHBoxLayout()
        self.__label_img_name = QLabel("Image Name")
        # self.__label_img_name.setText("Image Name")
        self.__label_img_name.setFixedHeight(20)
        ssub_layout.addWidget(self.__label_img_name)
        self.__edit_img_name = QLineEdit()
        self.__edit_img_name.setFixedHeight(20)
        self.__edit_img_name.setReadOnly(True)
        ssub_layout.addWidget(self.__edit_img_name)
        sub_layout.addLayout(ssub_layout)

        sub_layout.addWidget(QSplitter())  # 占位符

        # labeled fingerprint pose
        ssub_layout = QHBoxLayout()
        self.__label_fp_pose = QLabel("FP Pose")
        # self.__label_fp_pose.setText("FP Pose")
        self.__label_fp_pose.setFixedHeight(20)
        ssub_layout.addWidget(self.__label_fp_pose)
        self.__edit_center_x = QLineEdit()
        self.__edit_center_x.setFixedWidth(50)
        self.__edit_center_x.setFixedHeight(20)
        self.__edit_center_x.setReadOnly(True)
        ssub_layout.addWidget(self.__edit_center_x)
        self.__edit_center_y = QLineEdit()
        self.__edit_center_y.setFixedWidth(50)
        self.__edit_center_y.setFixedHeight(20)
        self.__edit_center_y.setReadOnly(True)
        ssub_layout.addWidget(self.__edit_center_y)
        self.__edit_center_angle = QLineEdit()
        self.__edit_center_angle.setFixedWidth(50)
        self.__edit_center_angle.setFixedHeight(20)
        self.__edit_center_angle.setReadOnly(True)
        ssub_layout.addWidget(self.__edit_center_angle)
        sub_layout.addLayout(ssub_layout)

        # prev or next button
        ssub_layout = QHBoxLayout()
        self.__btn_Prev = QPushButton("Prev")
        ssub_layout.addWidget(self.__btn_Prev)
        self.__btn_Next = QPushButton("Next")
        ssub_layout.addWidget(self.__btn_Next)
        sub_layout.addLayout(ssub_layout)

        # save buttion
        self.__btn_Save = QPushButton("Save")
        sub_layout.addWidget(self.__btn_Save)

        main_layout.addLayout(sub_layout)

        top_layout.addLayout(main_layout)

    @property
    def cur_pose(self):
        return self.__cur_pose

    @cur_pose.setter
    def cur_pose(self, value):
        # update when value changed
        if np.any(self.__cur_pose != value):
            self.__edit_center_x.setText(f"{np.rint(value[0] / self.__scale_ratio):.0f}")
            self.__edit_center_y.setText(f"{np.rint(value[1] / self.__scale_ratio):.0f}")
            self.__edit_center_angle.setText(f"{value[2]:.2f}")
            self.__cur_pose = value
            self.__is_modifed = True

    def __SetupConnection(self):
        self.__btn_Prev.clicked.connect(self.on_btn_Prev_Clicked)
        self.__btn_Next.clicked.connect(self.on_btn_Next_Clicked)
        self.__btn_Save.clicked.connect(self.on_btn_Save_Clicked)
        self.__img_box.ready.connect(self.on_pose_Changed)

    def on_btn_Prev_Clicked(self):
        if self.__is_modifed:
            self.on_btn_Save_Clicked()

        self.__index = (self.__index - 1) % len(self.__name_lst)
        self.fetch_new_data()

    def on_btn_Next_Clicked(self):
        if self.__is_modifed:
            self.on_btn_Save_Clicked()

        self.__index = (self.__index + 1) % len(self.__name_lst)
        self.fetch_new_data()

    def on_btn_Save_Clicked(self):
        if not self.__is_modifed or np.all(self.cur_pose == 0):
            # QMessageBox.warning(self, "Warning", "Initialized pose has not been changed")
            return

        try:
            img_name = self.__name_lst[self.__index]
            fpath = osp.join(self.__pose_prefix, f"{img_name}.txt")
            if not osp.exists(osp.dirname(fpath)):
                os.makedirs(osp.dirname(fpath))
            with open(fpath, "w") as fp:
                x1 = np.rint(self.cur_pose[0] / self.__scale_ratio)
                y1 = np.rint(self.cur_pose[1] / self.__scale_ratio)
                fp.write(f"{x1:.0f} {y1:.0f} {self.cur_pose[2]:.2f}")
            self.__label_info.setText(f"Save pose to {img_name} done")
        except Exception as ex:
            self.__label_info.setText(ex)
        self.__is_modifed = False

    def on_pose_Changed(self, pose_pts):
        x1, y1 = pose_pts[0]
        x2, y2 = pose_pts[1]
        self.cur_pose = np.array([x1, y1, np.rad2deg(np.arctan2(x1 - x2, y1 - y2))])

    def fetch_new_data(self):
        img_name = self.__name_lst[self.__index]

        self.__edit_img_name.setText(img_name)
        self.draw_image(osp.join(self.__prefix, f"{img_name}.{self.__ext}"))
        if osp.exists(osp.join(self.__pose_prefix, f"{img_name}.txt")):
            tmp = np.loadtxt(osp.join(self.__pose_prefix, f"{img_name}.txt"))
            tmp[:2] = np.rint(tmp[:2] * self.__scale_ratio)
            self.cur_pose = tmp
        else:
            self.cur_pose = np.zeros(3)
        x1, y1 = self.cur_pose[:2]
        x2 = x1 - 100 * np.sin(np.deg2rad(self.cur_pose[2]))
        y2 = y1 - 100 * np.cos(np.deg2rad(self.cur_pose[2]))
        self.__img_box.draw_pose((x1, y1), (x2, y2))
        self.__is_modifed = False

    def draw_image(self, path):
        pixmap = QPixmap(path)
        img_width = pixmap.width()
        img_height = pixmap.height()

        margin = 20
        if img_width >= img_height:
            self.__scale_ratio = (self.__img_panel_width - margin) * 1.0 / img_width
            pixmap = pixmap.scaledToWidth(self.__img_panel_width - margin)
        else:
            self.__scale_ratio = (self.__img_panel_height - margin) * 1.0 / img_height
            pixmap = pixmap.scaledToHeight(self.__img_panel_height - margin)

        self.__img_box.setPixmap(pixmap)

    def Quit(self):
        self.close()


if __name__ == "__main__":
    # Just an example, import this class in your own project, DO NOT MODIFY THIS CODE!
    app = QApplication(sys.argv)

    prefix = "/home/duanyongjie/data/finger/ContactSerials/Hefei/rolled"
    name_filter = "*"
    main_win = MainWidget(prefix=prefix, name_filter=name_filter, ext="png")
    main_win.show()

    exit(app.exec_())
