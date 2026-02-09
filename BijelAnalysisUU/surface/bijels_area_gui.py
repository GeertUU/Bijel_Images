import os
import sys
from typing import Optional, Tuple

import numpy as np
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets

from surface.bijels_area_estimation import (
    ProcessingParams,
    load_image,
    process_image,
    save_image,
)


def cv_bgr_to_qimage(bgr: np.ndarray) -> QtGui.QImage:
    if bgr is None:
        return QtGui.QImage()
    h, w, ch = bgr.shape
    bytes_per_line = ch * w
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)


def cv_gray_to_qimage(gray: np.ndarray) -> QtGui.QImage:
    if gray is None:
        return QtGui.QImage()
    h, w = gray.shape
    return QtGui.QImage(gray.data, w, h, w, QtGui.QImage.Format_Grayscale8)


def make_pixmap_from_cv(img: np.ndarray) -> QtGui.QPixmap:
    if img.ndim == 2:
        qimg = cv_gray_to_qimage(img)
    else:
        qimg = cv_bgr_to_qimage(img)
    return QtGui.QPixmap.fromImage(qimg)


def plot_hist_on_pixmap(hist: np.ndarray, size: Tuple[int, int] = (400, 150)) -> QtGui.QPixmap:
    w, h = size
    pix = QtGui.QPixmap(w, h)
    pix.fill(QtCore.Qt.white)
    painter = QtGui.QPainter(pix)
    painter.setPen(QtGui.QPen(QtCore.Qt.black))
    if hist is not None and len(hist) > 0:
        max_val = float(np.max(hist)) if np.max(hist) > 0 else 1.0
        bin_w = w / 256.0
        for i in range(256):
            val = float(hist[i]) / max_val
            bar_h = int(val * (h - 10))
            painter.drawLine(int(i * bin_w), h - 1, int(i * bin_w), h - 1 - bar_h)
    painter.end()
    return pix

def label_hist_pixmap(label: QtWidgets.QLabel, hist: np.ndarray, default_height: int = 150) -> QtGui.QPixmap:
    # Create histogram pixmap with width matching the display label
    target_w = max(label.width(), 200)
    return plot_hist_on_pixmap(hist, (target_w, default_height))


class ImagePanel(QtWidgets.QWidget):
    def __init__(self, title: str):
        super().__init__()
        self._orig_pixmap: Optional[QtGui.QPixmap] = None
        self.title_label = QtWidgets.QLabel(title)
        self.title_label.setStyleSheet("font-weight: bold;")
        self.top_label = QtWidgets.QLabel()
        self.top_label.setAlignment(QtCore.Qt.AlignCenter)
        self.hist_label = QtWidgets.QLabel()
        # Keep horizontally centered under the image but top-aligned vertically
        self.hist_label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter)
        self.bottom_widget = QtWidgets.QWidget()
        self.bottom_layout = QtWidgets.QVBoxLayout(self.bottom_widget)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.title_label)
        layout.addWidget(self.top_label, stretch=3)
        layout.addWidget(self.hist_label, stretch=1)
        layout.addWidget(self.bottom_widget)

    def set_top_pixmap(self, pixmap: QtGui.QPixmap):
        self._orig_pixmap = pixmap
        self._apply_scaled_pixmap()

    def _apply_scaled_pixmap(self):
        if self._orig_pixmap and not self._orig_pixmap.isNull():
            target_size = self.top_label.size()
            if target_size.width() > 0 and target_size.height() > 0:
                scaled = self._orig_pixmap.scaled(
                    target_size,
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
                self.top_label.setPixmap(scaled)
        else:
            self.top_label.clear()

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        self._apply_scaled_pixmap()


class BijelsAreaApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bijels Area Estimation")
        self.resize(1300, 900)

        self.image_path: Optional[str] = None
        self.input_bgr: Optional[np.ndarray] = None
        self.params = ProcessingParams()
        self.results = None
        # Pixel resolution in micrometers (µm)
        self.resolution_nm: float = 0.1202
        # Ensure default overlay color is red (BGR)
        self.params.overlay_color = (0, 0, 255)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        vroot = QtWidgets.QVBoxLayout(central)
        vroot.setSpacing(8)

        # Create panels
        self.panel_input = ImagePanel("1. Input Image")
        self.panel_eq = ImagePanel("2. Equalized Image")
        self.panel_smooth = ImagePanel("3. Smoothen Image")
        self.panel_quant = ImagePanel("4. Quantized Image")
        self.panel_final = ImagePanel("5. Final Image")

        # Top row: panels (image + histogram) side-by-side
        images_row = QtWidgets.QHBoxLayout()
        images_row.setSpacing(10)
        images_row.addWidget(self.panel_input, stretch=1)
        images_row.addWidget(self.panel_eq, stretch=1)
        images_row.addWidget(self.panel_smooth, stretch=1)
        images_row.addWidget(self.panel_quant, stretch=1)
        images_row.addWidget(self.panel_final, stretch=1)
        vroot.addLayout(images_row, stretch=3)

        # Bottom row: additional info/controls aligned below images
        info_row = QtWidgets.QHBoxLayout()
        info_row.setSpacing(10)

        # Build bottom widgets for each panel
        self._setup_panel_input_bottom()
        info_row.addWidget(self.panel_input.bottom_widget, stretch=1, alignment=QtCore.Qt.AlignTop)

        self._setup_panel_eq_bottom()
        info_row.addWidget(self.panel_eq.bottom_widget, stretch=1, alignment=QtCore.Qt.AlignTop)

        self._setup_panel_smooth_bottom()
        info_row.addWidget(self.panel_smooth.bottom_widget, stretch=1, alignment=QtCore.Qt.AlignTop)

        self._setup_panel_quant_bottom()
        info_row.addWidget(self.panel_quant.bottom_widget, stretch=1, alignment=QtCore.Qt.AlignTop)

        self._setup_panel_final_bottom()
        info_row.addWidget(self.panel_final.bottom_widget, stretch=1, alignment=QtCore.Qt.AlignTop)

        vroot.addLayout(info_row, stretch=2)

        self._setup_menu()
        self._update_ui()

    def _setup_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        load_act = QtWidgets.QAction("Load Image", self)
        load_act.triggered.connect(self.on_load_image)
        file_menu.addAction(load_act)
        save_act = QtWidgets.QAction("Save Final Image", self)
        save_act.triggered.connect(self.on_save_final)
        file_menu.addAction(save_act)

    def _setup_panel_input_bottom(self):
        self.input_info_label = QtWidgets.QLabel("No image loaded")
        self.panel_input.bottom_layout.addWidget(self.input_info_label)

    def _setup_panel_eq_bottom(self):
        self.eq_checkbox = QtWidgets.QCheckBox("Use Histogram Equalization")
        self.eq_checkbox.setChecked(self.params.use_hist_eq)
        self.eq_checkbox.stateChanged.connect(self.on_params_changed)
        self.panel_eq.bottom_layout.addWidget(self.eq_checkbox)

    def _setup_panel_smooth_bottom(self):
        self.kernel_combo = QtWidgets.QComboBox()
        for k in [2, 3, 4, 5, 6, 7]:
            self.kernel_combo.addItem(f"{k}x{k}", (k, k))
        # Default to 6x6 (index of value 6 in the list is 4)
        self.kernel_combo.setCurrentIndex(4)
        self.kernel_combo.currentIndexChanged.connect(self.on_params_changed)

        # Sigma X controls: slider + value box (two-way binding)
        self.sigma_x_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sigma_x_slider.setRange(0, 50)  # 0.0 to 5.0 in steps of 0.1
        self.sigma_x_slider.setValue(max(0, min(50, int(self.params.sigma_x * 10))))
        self.sigma_x_slider.valueChanged.connect(self.on_params_changed)
        self.sigma_x_value = QtWidgets.QLineEdit(f"{self.sigma_x_slider.value()/10.0:.1f}")
        self.sigma_x_value.setMaximumWidth(60)
        self.sigma_x_value.setValidator(QtGui.QDoubleValidator(0.0, 5.0, 1))
        self.sigma_x_value.editingFinished.connect(self._on_sigma_x_text_changed)

        # Sigma Y controls: slider + value box (two-way binding)
        self.sigma_y_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sigma_y_slider.setRange(0, 50)  # 0.0 to 5.0 in steps of 0.1
        self.sigma_y_slider.setValue(max(0, min(50, int(self.params.sigma_y * 10))))
        self.sigma_y_slider.valueChanged.connect(self.on_params_changed)
        self.sigma_y_value = QtWidgets.QLineEdit(f"{self.sigma_y_slider.value()/10.0:.1f}")
        self.sigma_y_value.setMaximumWidth(60)
        self.sigma_y_value.setValidator(QtGui.QDoubleValidator(0.0, 5.0, 1))
        self.sigma_y_value.editingFinished.connect(self._on_sigma_y_text_changed)

        self.gaussian_checkbox = QtWidgets.QCheckBox("Use Gaussian Smoothing")
        self.gaussian_checkbox.setChecked(self.params.use_gaussian)
        self.gaussian_checkbox.stateChanged.connect(self.on_params_changed)

        form = QtWidgets.QFormLayout()
        form.addRow("Kernel Size", self.kernel_combo)
        sx_row = QtWidgets.QHBoxLayout()
        sx_row.addWidget(self.sigma_x_slider)
        sx_row.addWidget(self.sigma_x_value)
        sy_row = QtWidgets.QHBoxLayout()
        sy_row.addWidget(self.sigma_y_slider)
        sy_row.addWidget(self.sigma_y_value)
        form.addRow("Sigma X (0-5)", self._wrap_layout_widget(sx_row))
        form.addRow("Sigma Y (0-5)", self._wrap_layout_widget(sy_row))
        form.addRow(self.gaussian_checkbox)
        self.panel_smooth.bottom_layout.addLayout(form)

    def _setup_panel_quant_bottom(self):
        self.quant_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.quant_slider.setRange(0, 255)
        self.quant_slider.setValue(self.params.quant_threshold)
        self.quant_slider.valueChanged.connect(self.on_params_changed)
        # Quant threshold value box (two-way binding)
        self.quant_value = QtWidgets.QLineEdit(str(self.quant_slider.value()))
        self.quant_value.setMaximumWidth(60)
        self.quant_value.setValidator(QtGui.QIntValidator(0, 255))
        self.quant_value.editingFinished.connect(self._on_quant_text_changed)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.quant_slider)
        row.addWidget(self.quant_value)
        self.panel_quant.bottom_layout.addWidget(QtWidgets.QLabel("Quantization Threshold"))
        self.panel_quant.bottom_layout.addWidget(self._wrap_layout_widget(row))

    def _setup_panel_final_bottom(self):
        self.update_button = QtWidgets.QPushButton("Update Final")
        self.update_button.clicked.connect(self.on_params_changed)
        self.resolution_input = QtWidgets.QLineEdit("0.1202")
        self.resolution_input.setValidator(QtGui.QDoubleValidator(0.0, 1e9, 6))
        self.resolution_input.editingFinished.connect(self.on_params_changed)
        # Overlay choice: choose which mask value to overlay
        self.overlay_zero_radio = QtWidgets.QRadioButton("Overlay mask==0")
        self.overlay_255_radio = QtWidgets.QRadioButton("Overlay mask==255")
        self.overlay_255_radio.setChecked(True)
        self.overlay_zero_radio.toggled.connect(self.on_params_changed)
        # Color selector for overlay
        self.overlay_color_button = QtWidgets.QPushButton("Overlay Color")
        self.overlay_color_button.clicked.connect(self._choose_overlay_color)
        # Show current color swatch
        self.overlay_color_swatch = QtWidgets.QLabel()
        self.overlay_color_swatch.setFixedSize(24, 24)
        self._update_overlay_swatch()
        self.stats_label = QtWidgets.QLabel("Stats: N/A")
        form = QtWidgets.QFormLayout()
        form.addRow(self.update_button)
        form.addRow("Pixel Resolution (µm)", self.resolution_input)
        overlay_row = QtWidgets.QHBoxLayout()
        overlay_row.addWidget(self.overlay_zero_radio)
        overlay_row.addWidget(self.overlay_255_radio)
        form.addRow("Overlay Pixels", self._wrap_layout_widget(overlay_row))
        color_row = QtWidgets.QHBoxLayout()
        color_row.addWidget(self.overlay_color_button)
        color_row.addWidget(self.overlay_color_swatch)
        form.addRow("Overlay Color", self._wrap_layout_widget(color_row))
        form.addRow("Image Stats", self.stats_label)
        self.panel_final.bottom_layout.addLayout(form)

    def on_load_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not path:
            return
        img = load_image(path)
        if img is None:
            QtWidgets.QMessageBox.warning(self, "Load Failed", "Could not load image.")
            return
        self.image_path = path
        self.input_bgr = img
        self._run_pipeline()
        self._update_ui()

    def on_save_final(self):
        if self.results is None:
            QtWidgets.QMessageBox.information(self, "No Result", "Please load an image and update.")
            return
        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Final Image", "final_overlay.png", "PNG (*.png);;JPEG (*.jpg *.jpeg)"
        )
        if not save_path:
            return
        final_overlay = self.results[5]
        ok = save_image(save_path, final_overlay)
        if ok:
            QtWidgets.QMessageBox.information(self, "Saved", f"Saved to: {save_path}")
        else:
            QtWidgets.QMessageBox.warning(self, "Save Failed", "Could not save image.")

    def on_params_changed(self):
        self.params.use_hist_eq = self.eq_checkbox.isChecked()
        self.params.use_gaussian = self.gaussian_checkbox.isChecked()
        self.params.gaussian_kernel = self.kernel_combo.currentData()
        self.params.sigma_x = self.sigma_x_slider.value() / 10.0
        self.params.sigma_y = self.sigma_y_slider.value() / 10.0
        # Clamp to [0, 5]
        self.params.sigma_x = max(0.0, min(5.0, self.params.sigma_x))
        self.params.sigma_y = max(0.0, min(5.0, self.params.sigma_y))
        # Reflect slider changes to text boxes
        self.sigma_x_value.setText(f"{self.params.sigma_x:.1f}")
        self.sigma_y_value.setText(f"{self.params.sigma_y:.1f}")
        self.params.quant_threshold = self.quant_slider.value()
        self.quant_value.setText(str(self.params.quant_threshold))
        # Overlay mode: True for zeros, False for 255
        self.params.overlay_use_zero = self.overlay_zero_radio.isChecked()
        
        try:
            self.resolution_nm = float(self.resolution_input.text())
        except Exception:
            self.resolution_nm = 1.0
        self._run_pipeline()
        self._update_ui()

    def _choose_overlay_color(self):
        # QColorDialog returns RGB; convert to BGR for OpenCV usage
        current_bgr = getattr(self.params, 'overlay_color', (0, 0, 255))
        current_rgb = QtGui.QColor(current_bgr[2], current_bgr[1], current_bgr[0])
        color = QtWidgets.QColorDialog.getColor(current_rgb, self, "Select Overlay Color")
        if color.isValid():
            r, g, b, _ = color.getRgb()
            self.params.overlay_color = (b, g, r)  # store as BGR
            self._update_overlay_swatch()
            self._run_pipeline()
            self._update_ui()

    def _update_overlay_swatch(self):
        # Update swatch label background to current overlay color
        bgr = getattr(self.params, 'overlay_color', (0, 0, 255))
        r, g, b = bgr[2], bgr[1], bgr[0]
        self.overlay_color_swatch.setStyleSheet(f"background-color: rgb({r}, {g}, {b}); border: 1px solid #666;")

    # Helpers for two-way binding
    def _wrap_layout_widget(self, layout: QtWidgets.QLayout) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        container.setLayout(layout)
        return container

    def _on_sigma_x_text_changed(self):
        try:
            val = float(self.sigma_x_value.text())
        except Exception:
            return
        val = max(0.0, min(5.0, val))
        self.sigma_x_slider.setValue(int(round(val * 10)))
        # Trigger pipeline update via on_params_changed
        self.on_params_changed()

    def _on_sigma_y_text_changed(self):
        try:
            val = float(self.sigma_y_value.text())
        except Exception:
            return
        val = max(0.0, min(5.0, val))
        self.sigma_y_slider.setValue(int(round(val * 10)))
        self.on_params_changed()

    def _on_quant_text_changed(self):
        try:
            val = int(self.quant_value.text())
        except Exception:
            return
        val = max(0, min(255, val))
        self.quant_slider.setValue(val)
        self.on_params_changed()

    def _run_pipeline(self):
        if self.input_bgr is None:
            self.results = None
            return
        self.results = process_image(self.input_bgr, self.params)

    def _update_ui(self):
        if self.input_bgr is not None and self.results is not None:
            input_color, gray, eq_gray, smooth_gray, quant_mask, final_overlay, stats = self.results

            self.panel_input.set_top_pixmap(make_pixmap_from_cv(input_color))
            input_hist_pix = label_hist_pixmap(self.panel_input.top_label, stats["hist_input_gray"]) 
            self.panel_input.hist_label.setPixmap(input_hist_pix)
            if self.image_path:
                h, w, _ = self.input_bgr.shape
                info = f"File: {os.path.basename(self.image_path)} | Size: {w} x {h}"
            else:
                info = "No image loaded"
            self.input_info_label.setText(info)

            self.panel_eq.set_top_pixmap(make_pixmap_from_cv(eq_gray))
            eq_hist_pix = label_hist_pixmap(self.panel_eq.top_label, stats["hist_eq_gray"]) 
            self.panel_eq.hist_label.setPixmap(eq_hist_pix)

            self.panel_smooth.set_top_pixmap(make_pixmap_from_cv(smooth_gray))
            self.panel_quant.set_top_pixmap(make_pixmap_from_cv(quant_mask))

            self.panel_final.set_top_pixmap(make_pixmap_from_cv(final_overlay))
            total_pixels = stats["total_pixels"]
            count_0 = stats.get("count_0", 0)
            percent_0 = stats.get("percent_0", (count_0 / total_pixels * 100.0) if total_pixels else 0.0)
            # Derive 255 stats if not provided
            count_255 = total_pixels - count_0
            percent_255 = (count_255 / total_pixels * 100.0) if total_pixels else 0.0

            # Use overlay selection to drive displayed stats and area
            overlay_zeros = self.params.overlay_use_zero
            selected_label = "Quantized 0" if overlay_zeros else "Quantized 255"
            selected_count = count_0 if overlay_zeros else count_255
            selected_percent = percent_0 if overlay_zeros else percent_255
            surface_area_um2 = selected_count * (self.resolution_nm ** 2)
            stats_text = (
                f"{selected_label}: {selected_count} | %: {selected_percent:.2f}% | "
                f"Surface area: {surface_area_um2:.4f} µm^2"
            )
            self.stats_label.setText(stats_text)
        else:
            self.panel_input.set_top_pixmap(QtGui.QPixmap())
            self.panel_input.hist_label.clear()
            self.input_info_label.setText("No image loaded")
            self.panel_eq.set_top_pixmap(QtGui.QPixmap())
            self.panel_eq.hist_label.clear()
            self.panel_smooth.set_top_pixmap(QtGui.QPixmap())
            self.panel_quant.set_top_pixmap(QtGui.QPixmap())
            self.panel_final.set_top_pixmap(QtGui.QPixmap())
            self.stats_label.setText("Stats: N/A")


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = BijelsAreaApp()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
