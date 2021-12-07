"""
This file (manual_exclude_imgs.py.py) is designed for:
    manual select images to exclude, and save image names in specific file
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
    QRadioButton,
    QSplitter,
    QLineEdit,
    QLabel,
    QFrame,
    QMessageBox,
)
from PyQt5.QtGui import QIntValidator, QImage, QPixmap, QPainter, QPen
from PyQt5 import QtCore
from pyqtgraph.widgets.RawImageWidget import RawImageWidget


class MainWidget(QWidget):
    def __init__(self, prefix, name_filter="*", ext="png", Parent=None) -> None:
        """Constructor"""
        super().__init__(Parent)

        self.__prefix = prefix
        self.__name_filter = name_filter
        self.__ext = ext
        self.__exclude_fpath = prefix + "_exclude.txt"

        self.__name_lst = glob(osp.join(prefix, f"**/{name_filter}.{ext}"), recursive=True)
        self.__name_lst = [x.replace(f"{prefix}/", "").replace(f".{ext}", "") for x in self.__name_lst]
        self.__name_lst.sort()

        if len(self.__name_lst) == 0:
            QMessageBox.critical(self, "Error", "No image file with specified format is found")
            return

        self.__img_panel_width = 640
        self.__img_panel_height = 640

        self.__InitView()
        self.__SetupConnection()

        try:
            with open(self.__exclude_fpath, "r") as fp:
                self.__exclude_lst = fp.read().splitlines()
        except Exception as ex:
            self.__exclude_lst = []
            self.show_info(f"Error: {ex}")

        self.__n_imgs = len(self.__name_lst)
        self.__index = -1
        self.on_btn_Next_Clicked()

    def __InitView(self):
        """初始化界面"""
        self.setWindowTitle("Exclusion Selection")

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
        self.__img_box = QLabel()
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

        # if excluding
        self.__rad_Flag = QRadioButton("Exclude", self)
        self.__rad_Flag.setChecked(False)
        sub_layout.addWidget(self.__rad_Flag)

        sub_layout.addWidget(QSplitter())  # 占位符

        # prev or next button
        ssub_layout = QHBoxLayout()
        self.__btn_Prev = QPushButton("Prev")
        ssub_layout.addWidget(self.__btn_Prev)
        self.__btn_Next = QPushButton("Next")
        ssub_layout.addWidget(self.__btn_Next)
        sub_layout.addLayout(ssub_layout)

        # add, del, and clear button
        self.__btn_Clear = QPushButton("Clear", self)
        sub_layout.addWidget(self.__btn_Clear)

        # save buttion
        self.__btn_Save = QPushButton("Save")
        sub_layout.addWidget(self.__btn_Save)

        main_layout.addLayout(sub_layout)

        top_layout.addLayout(main_layout)

    def __SetupConnection(self):
        self.__btn_Prev.clicked.connect(self.on_btn_Prev_Clicked)
        self.__btn_Next.clicked.connect(self.on_btn_Next_Clicked)
        self.__rad_Flag.toggled.connect(self.on_rad_Flag_Toggled)
        self.__btn_Clear.clicked.connect(self.on_btn_Clear_Clicked)
        self.__btn_Save.clicked.connect(self.on_btn_Save_Clicked)

    def on_btn_Prev_Clicked(self):
        self.__index = (self.__index - 1) % len(self.__name_lst)

        self.fetch_new_data()

    def on_btn_Next_Clicked(self):
        self.__index = (self.__index + 1) % len(self.__name_lst)

        self.fetch_new_data()

    def on_rad_Flag_Toggled(self):
        img_name = self.__name_lst[self.__index]
        if self.__rad_Flag.isChecked():
            if img_name not in self.__exclude_lst:
                self.__exclude_lst.append(img_name)
                self.show_info(f"Add {img_name} to excluding list")
            else:
                self.show_info("Already excluded")
        else:
            if img_name in self.__exclude_lst:
                self.__exclude_lst.remove(img_name)
                self.show_info(f"Delete {img_name} from excluding list")
            else:
                self.show_info("Not Exist")

    def on_btn_Clear_Clicked(self):
        try:
            with open(self.__exclude_fpath, "r") as fp:
                self.__exclude_lst = fp.read().splitlines()
        except Exception as ex:
            self.__exclude_lst = []
            self.show_info(f"Error: {ex}")

        self.__rad_Flag.setChecked(self.__name_lst[self.__index] in self.__exclude_lst)

    def on_btn_Save_Clicked(self):
        try:
            with open(self.__exclude_fpath, "w") as fp:
                fp.write("\n".join(self.__exclude_lst))
            self.show_info(f"Exclude files saved to {self.__exclude_fpath}")
        except Exception as ex:
            self.show_info(f"Error: {ex}")

    def fetch_new_data(self):
        img_name = self.__name_lst[self.__index]

        self.__edit_img_name.setText(f"[{self.__index}/{self.__n_imgs}] {img_name}")
        self.draw_image(osp.join(self.__prefix, f"{img_name}.{self.__ext}"))

        self.__rad_Flag.setChecked(img_name in self.__exclude_lst)

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

    def show_info(self, info):
        self.__label_info.setText(info)

    def Quit(self):
        self.close()


if __name__ == "__main__":
    # Just an example, import this class in your own project, DO NOT MODIFY THIS CODE!
    app = QApplication(sys.argv)

    prefix = "/mnt/data1/hefei_data/processed/dyj_finger_contact/align/mosaic_plain"
    name_filter = "*"
    main_win = MainWidget(prefix=prefix, name_filter=name_filter, ext="png")
    main_win.show()

    exit(app.exec_())
