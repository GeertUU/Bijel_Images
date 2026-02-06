import sys
import os
from typing import List, Optional

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

# Helper: read multi-page TIFF using OpenCV if possible; fallback to PIL
try:
    from PIL import Image
except ImportError:
    Image = None


def imread_multipage_tiff(path: str) -> List[np.ndarray]:
    images: List[np.ndarray] = []
    # OpenCV's imreadmulti works for many TIFFs
    try:
        ok, pages = cv2.imreadmulti(path, flags=cv2.IMREAD_UNCHANGED)
        if ok and pages:
            images = pages
    except Exception:
        pass
    if (not images) and Image is not None:
        try:
            with Image.open(path) as img:
                for i in range(img.n_frames):
                    img.seek(i)
                    arr = np.array(img)
                    # Convert grayscale or RGBA to BGR for uniform display
                    if arr.ndim == 2:
                        bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                    elif arr.ndim == 3 and arr.shape[2] == 4:
                        bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
                    elif arr.ndim == 3 and arr.shape[2] == 3:
                        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                    else:
                        bgr = arr if arr.ndim == 3 else cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                    images.append(bgr)
        except Exception:
            pass
    return images


def cv_bgr_to_qimage(bgr: np.ndarray) -> QtGui.QImage:
    if bgr is None:
        return QtGui.QImage()
    h, w = bgr.shape[:2]
    if bgr.ndim == 2:
        return QtGui.QImage(bgr.data, w, h, w, QtGui.QImage.Format_Grayscale8)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    bytes_per_line = 3 * w
    return QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)


def make_pixmap(img: np.ndarray) -> QtGui.QPixmap:
    return QtGui.QPixmap.fromImage(cv_bgr_to_qimage(img))


class ImageViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TIF Viewer")
        self.resize(1000, 700)

        self.images: List[np.ndarray] = []
        self.current_index: int = 0
        self.fit_to_window: bool = False
        self.current_pixmap: Optional[QtGui.QPixmap] = None
        self.current_path: Optional[str] = None

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        vroot = QtWidgets.QVBoxLayout(central)
        vroot.setContentsMargins(6, 6, 6, 6)
        vroot.setSpacing(6)

        # Scroll area for image
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.scroll_area.setWidget(self.image_label)
        vroot.addWidget(self.scroll_area, stretch=1)

        # Slider + textbox under image
        controls_container = QtWidgets.QWidget()
        controls_layout = QtWidgets.QHBoxLayout(controls_container)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)
        self.page_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.page_slider.setRange(0, 0)  # will update when images loaded
        self.page_slider.valueChanged.connect(self.on_slider_changed)
        self.page_edit = QtWidgets.QLineEdit("0")
        self.page_edit.setMaxLength(7)
        self.page_edit.setFixedWidth(80)
        self.page_edit.setValidator(QtGui.QIntValidator(0, 10**7 - 1))
        self.page_edit.editingFinished.connect(self.on_edit_changed)
        controls_layout.addWidget(self.page_slider, stretch=1)
        controls_layout.addWidget(self.page_edit)
        vroot.addWidget(controls_container)

        self._setup_menu()
        self._update_controls_enabled(False)

    def _setup_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        open_act = QtWidgets.QAction("Open File", self)
        open_act.triggered.connect(self.on_open)
        file_menu.addAction(open_act)
        save_act = QtWidgets.QAction("Save Image", self)
        save_act.triggered.connect(self.on_save)
        file_menu.addAction(save_act)

        view_menu = menubar.addMenu("View")
        fit_act = QtWidgets.QAction("Fit Image to Window", self)
        fit_act.setCheckable(True)
        fit_act.triggered.connect(self.on_toggle_fit)
        view_menu.addAction(fit_act)
        self.fit_action = fit_act

    def _update_controls_enabled(self, enabled: bool):
        self.page_slider.setEnabled(enabled)
        self.page_edit.setEnabled(enabled)

    def on_open(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open TIFF", "", "TIFF (*.tif *.tiff);;Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not path:
            return
        images = []
        if path.lower().endswith((".tif", ".tiff")):
            images = imread_multipage_tiff(path)
        else:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.ndim == 3 and img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                images = [img]
        if not images:
            QtWidgets.QMessageBox.warning(self, "Open Failed", "Could not open image.")
            return
        self.images = images
        self.current_path = path
        self.current_index = 0
        self.page_slider.setRange(0, len(self.images) - 1)
        self.page_slider.setValue(0)
        self.page_edit.setText(str(0))
        self._update_controls_enabled(True)
        self._show_current_image()

    def _show_current_image(self):
        if not self.images:
            self.image_label.clear()
            self.current_pixmap = None
            return
        bgr = self.images[self.current_index]
        pix = make_pixmap(bgr)
        self.current_pixmap = pix
        if self.fit_to_window:
            # Scale to fit inside scroll area viewport while keeping aspect ratio
            viewport_size = self.scroll_area.viewport().size()
            if not pix.isNull() and viewport_size.width() > 0 and viewport_size.height() > 0:
                scaled = pix.scaled(viewport_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                self.image_label.setPixmap(scaled)
            else:
                self.image_label.setPixmap(pix)
        else:
            self.image_label.setPixmap(pix)
        # Ensure scrollbars show when larger than viewport
        self.image_label.adjustSize()
        # Resize slider to roughly match image label width
        self._resize_controls_to_image_width()

    def _resize_controls_to_image_width(self):
        # Adjust the slider maximum width to match the displayed image width
        label_width = self.image_label.pixmap().width() if self.image_label.pixmap() else self.image_label.width()
        self.page_slider.setMaximumWidth(label_width)

    def on_slider_changed(self, value: int):
        if not self.images:
            return
        value = max(0, min(len(self.images) - 1, value))
        if value != self.current_index:
            self.current_index = value
            self.page_edit.setText(str(value))
            self._show_current_image()
        else:
            # Still update textbox in case user dragged but stayed same
            self.page_edit.setText(str(value))

    def on_edit_changed(self):
        if not self.images:
            return
        try:
            value = int(self.page_edit.text())
        except Exception:
            return
        value = max(0, min(len(self.images) - 1, value))
        self.page_slider.setValue(value)
        # _show_current_image will be called via slider change

    def on_toggle_fit(self, checked: bool):
        self.fit_to_window = checked
        self._show_current_image()

    def on_save(self):
        if not self.images:
            QtWidgets.QMessageBox.information(self, "No Image", "Load an image first.")
            return
        filters = "TIFF (*.tif *.tiff);;JPEG (*.jpg *.jpeg);;PNG (*.png)"
        # Build default file name: <original base>_<page:04d> with original extension if TIFF, else default to .tif
        base_name = "image"
        default_ext = ".tif"
        if self.current_path:
            base, ext = os.path.splitext(os.path.basename(self.current_path))
            base_name = base
            # prefer original extension if it is tif/tiff; otherwise keep .tif
            if ext.lower() in (".tif", ".tiff"):
                default_ext = ext
        default_name = f"{base_name}_{self.current_index:04d}{default_ext}"
        save_path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(self, "Save Image", default_name, filters)
        if not save_path:
            return
        # Get displayed image pixels (use current original BGR for fidelity)
        bgr = self.images[self.current_index]
        ext = os.path.splitext(save_path)[1].lower()
        params = []
        if ext in (".jpg", ".jpeg"):
            params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        try:
            ok = cv2.imwrite(save_path, bgr, params)
        except Exception:
            ok = False
        if ok:
            QtWidgets.QMessageBox.information(self, "Saved", f"Saved to: {save_path}")
        else:
            QtWidgets.QMessageBox.warning(self, "Save Failed", "Could not save image.")


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = ImageViewer()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
