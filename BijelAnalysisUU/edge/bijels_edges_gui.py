import os
import sys
from typing import Optional

import numpy as np
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets

from edge.bijels_edges_estimation import (
    ProcessingParams,
    EDGE_ALGOS,
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


def plot_hist_on_pixmap(hist: np.ndarray, size=(300, 120)) -> QtGui.QPixmap:
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


class Panel(QtWidgets.QWidget):
    def __init__(self, title: str):
        super().__init__()
        self._orig_pixmap: Optional[QtGui.QPixmap] = None
        self.title = QtWidgets.QLabel(title)
        self.title.setStyleSheet("font-weight: bold;")
        self.top = QtWidgets.QLabel()
        self.top.setAlignment(QtCore.Qt.AlignCenter)
        # Ensure top image area has a consistent height across panels
        self._image_min_height = 260
        sp_top = self.top.sizePolicy()
        sp_top.setHorizontalPolicy(QtWidgets.QSizePolicy.Preferred)
        sp_top.setVerticalPolicy(QtWidgets.QSizePolicy.Expanding)
        self.top.setSizePolicy(sp_top)
        self.top.setMinimumHeight(self._image_min_height)
        # Re-apply scaling when the internal image label resizes
        self.top.installEventFilter(self)
        self.bottom = QtWidgets.QWidget()
        self.bottom_layout = QtWidgets.QVBoxLayout(self.bottom)
        self.bottom_layout.setAlignment(QtCore.Qt.AlignTop)
        # Controls area: fixed height, width stretches with panel
        self._controls_fixed_height = 240
        sp = self.bottom.sizePolicy()
        sp.setHorizontalPolicy(QtWidgets.QSizePolicy.Preferred)
        sp.setVerticalPolicy(QtWidgets.QSizePolicy.Fixed)
        self.bottom.setSizePolicy(sp)
        self.bottom.setMinimumHeight(self._controls_fixed_height)
        self.bottom.setMaximumHeight(self._controls_fixed_height)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.title)
        layout.addWidget(self.top, stretch=3)
        layout.addWidget(self.bottom)

    def set_image(self, pix: QtGui.QPixmap):
        if pix and not pix.isNull():
            self._orig_pixmap = pix
            self._apply_scaled_pixmap()
        else:
            self._orig_pixmap = None
            self.top.clear()

    def _apply_scaled_pixmap(self):
        if self._orig_pixmap and not self._orig_pixmap.isNull():
            target_size = self.top.size()
            if target_size.width() > 0 and target_size.height() > 0:
                scaled = self._orig_pixmap.scaled(
                    target_size,
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
                self.top.setPixmap(scaled)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if obj is self.top and event.type() == QtCore.QEvent.Resize:
            self._apply_scaled_pixmap()
        return super().eventFilter(obj, event)

    def resizeEvent(self, e: QtGui.QResizeEvent):
        super().resizeEvent(e)
        # Ensure images scale to the available size and keep same visual size across panels
        self._apply_scaled_pixmap()


class BijelsEdgesApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bijels Edges Estimation")
        self.resize(1400, 900)

        self.params = ProcessingParams()
        self.image_path: Optional[str] = None
        self.input_bgr: Optional[np.ndarray] = None
        self.results = None

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        grid = QtWidgets.QGridLayout(central)
        grid.setSpacing(8)

        # Panels 3x2
        self.p_input = Panel("1. Input Image")
        self.p_eq = Panel("2. Equalized Image")
        self.p_smooth = Panel("3. Smoothen Image")
        self.p_edge = Panel("4. Edge Detection Image")
        self.p_quant = Panel("5. Quantized Image")
        self.p_final = Panel("6. Final Image")

        grid.addWidget(self.p_input, 0, 0)
        grid.addWidget(self.p_eq, 0, 1)
        grid.addWidget(self.p_smooth, 0, 2)
        grid.addWidget(self.p_edge, 1, 0)
        grid.addWidget(self.p_quant, 1, 1)
        grid.addWidget(self.p_final, 1, 2)

        self._setup_menu()
        self._setup_bottoms()
        self._update_ui()

    # Helper to embed a QLayout into a QWidget for use in FormLayouts
    def _wrap_layout_widget(self, layout: QtWidgets.QLayout) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        container.setLayout(layout)
        return container

    def _setup_menu(self):
        menubar = self.menuBar()
        f = menubar.addMenu("File")
        o = QtWidgets.QAction("Load Image", self)
        o.triggered.connect(self.on_load_image)
        f.addAction(o)
        s = QtWidgets.QAction("Save Final Image", self)
        s.triggered.connect(self.on_save_final)
        f.addAction(s)

    def _setup_bottoms(self):
        # 1. Input: image info only
        self.input_info = QtWidgets.QLabel("No image loaded")
        self.p_input.bottom_layout.addWidget(self.input_info)

        # 2. Equalized: checkbox only
        self.eq_checkbox = QtWidgets.QCheckBox("Use Histogram Equalization")
        self.eq_checkbox.setChecked(self.params.use_hist_eq)
        self.eq_checkbox.stateChanged.connect(self.on_params_changed)
        self.p_eq.bottom_layout.addWidget(self.eq_checkbox)

        # 3. Smoothen: controls
        self.kernel_combo = QtWidgets.QComboBox()
        for k in [2, 3, 4, 5, 6, 7]:
            self.kernel_combo.addItem(f"{k}x{k}", (k, k))
        # Default to 6x6
        idx_6 = [2, 3, 4, 5, 6, 7].index(6)
        self.kernel_combo.setCurrentIndex(idx_6)
        self.kernel_combo.currentIndexChanged.connect(self.on_params_changed)
        self.sigma_x = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sigma_x.setRange(0, 50)
        self.sigma_x.setValue(int(self.params.sigma_x * 10))
        self.sigma_x.valueChanged.connect(self.on_params_changed)
        self.sigma_x_edit = QtWidgets.QLineEdit()
        self.sigma_x_edit.setFixedWidth(60)
        self.sigma_x_edit.setValidator(QtGui.QDoubleValidator(0.0, 5.0, 2))
        self.sigma_x_edit.editingFinished.connect(self._on_sigma_x_edit)
        self.sigma_y = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sigma_y.setRange(0, 50)
        self.sigma_y.setValue(int(self.params.sigma_y * 10))
        self.sigma_y.valueChanged.connect(self.on_params_changed)
        self.sigma_y_edit = QtWidgets.QLineEdit()
        self.sigma_y_edit.setFixedWidth(60)
        self.sigma_y_edit.setValidator(QtGui.QDoubleValidator(0.0, 5.0, 2))
        self.sigma_y_edit.editingFinished.connect(self._on_sigma_y_edit)
        self.gauss_check = QtWidgets.QCheckBox("Use Gaussian Smoothing")
        # Default to checked (enabled)
        self.gauss_check.setChecked(True)
        self.gauss_check.stateChanged.connect(self.on_params_changed)
        form = QtWidgets.QFormLayout()
        form.addRow("Kernel Size", self.kernel_combo)
        sigma_x_row = QtWidgets.QHBoxLayout()
        sigma_x_row.addWidget(self.sigma_x)
        sigma_x_row.addWidget(self.sigma_x_edit)
        form.addRow("Sigma X (0-5)", self._wrap_layout_widget(sigma_x_row))
        sigma_y_row = QtWidgets.QHBoxLayout()
        sigma_y_row.addWidget(self.sigma_y)
        sigma_y_row.addWidget(self.sigma_y_edit)
        form.addRow("Sigma Y (0-5)", self._wrap_layout_widget(sigma_y_row))
        form.addRow(self.gauss_check)
        self.p_smooth.bottom_layout.addLayout(form)

        # 4. Edge Detection: selector
        self.edge_combo = QtWidgets.QComboBox()
        for a in EDGE_ALGOS:
            self.edge_combo.addItem(a)
        self.edge_combo.setCurrentText(self.params.edge_algo)
        self.edge_combo.currentIndexChanged.connect(self._on_edge_algo_changed)
        # Dynamic parameters area per algorithm
        self.edge_params_container = QtWidgets.QWidget()
        self.edge_params_layout = QtWidgets.QFormLayout(self.edge_params_container)
        # Initialize param controls
        self._init_edge_param_controls()
        self._rebuild_edge_params()
        edge_form = QtWidgets.QFormLayout()
        edge_form.addRow("Edge Algorithm", self.edge_combo)
        edge_form.addRow(self.edge_params_container)
        self.p_edge.bottom_layout.addLayout(edge_form)

        # 5. Quantized: slider
        self.quant_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.quant_slider.setRange(0, 255)
        self.quant_slider.setValue(self.params.quant_threshold)
        self.quant_slider.valueChanged.connect(self.on_params_changed)
        self.quant_edit = QtWidgets.QLineEdit()
        self.quant_edit.setFixedWidth(60)
        self.quant_edit.setValidator(QtGui.QIntValidator(0, 255))
        self.quant_edit.editingFinished.connect(self._on_quant_edit)
        self.p_quant.bottom_layout.addWidget(QtWidgets.QLabel("Quantization Threshold"))
        quant_row = QtWidgets.QHBoxLayout()
        quant_row.addWidget(self.quant_slider)
        quant_row.addWidget(self.quant_edit)
        self.p_quant.bottom_layout.addLayout(quant_row)

        # 6. Final: update button + resolution + stats
        self.update_btn = QtWidgets.QPushButton("Update Final")
        self.update_btn.clicked.connect(self.on_params_changed)
        self.res_input = QtWidgets.QLineEdit("0.2405")
        self.res_input.setValidator(QtGui.QDoubleValidator(0.0, 1e9, 6))
        self.res_input.editingFinished.connect(self.on_params_changed)
        # Overlay selection (match Bijels Area GUI)
        self.overlay_zero_radio = QtWidgets.QRadioButton("Overlay mask==0")
        self.overlay_255_radio = QtWidgets.QRadioButton("Overlay mask==255")
        self.overlay_255_radio.setChecked(True)
        self.overlay_zero_radio.toggled.connect(self.on_params_changed)
        # Overlay color button and swatch
        self.overlay_color_button = QtWidgets.QPushButton("Overlay Color")
        self.overlay_color_button.setMinimumHeight(36)
        self.overlay_color_button.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.overlay_color_button.clicked.connect(self._choose_overlay_color)
        self.overlay_color_swatch = QtWidgets.QLabel()
        self.overlay_color_swatch.setFixedSize(36, 24)
        self._update_overlay_swatch()
        self.stats_label = QtWidgets.QLabel("Stats: N/A")
        final_form = QtWidgets.QFormLayout()
        final_form.addRow(self.update_btn)
        final_form.addRow("Pixel Resolution (µm)", self.res_input)
        overlay_row = QtWidgets.QHBoxLayout()
        # Increase height of radio controls
        self.overlay_zero_radio.setMinimumHeight(36)
        self.overlay_255_radio.setMinimumHeight(36)
        overlay_row.addWidget(self.overlay_zero_radio)
        overlay_row.addWidget(self.overlay_255_radio)
        overlay_row_container = QtWidgets.QWidget()
        overlay_row_container.setMinimumHeight(44)
        overlay_row_container.setLayout(overlay_row)
        final_form.addRow("Overlay Pixels", overlay_row_container)
        color_row = QtWidgets.QHBoxLayout()
        color_row.addWidget(self.overlay_color_button)
        color_row.addWidget(self.overlay_color_swatch)
        color_row_container = QtWidgets.QWidget()
        color_row_container.setMinimumHeight(44)
        color_row_container.setLayout(color_row)
        final_form.addRow("Overlay Color", color_row_container)
        final_form.addRow("Image Stats", self.stats_label)
        self.p_final.bottom_layout.addLayout(final_form)

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
        self.params.use_gaussian = self.gauss_check.isChecked()
        self.params.gaussian_kernel = self.kernel_combo.currentData()
        # Keep edits in sync with sliders
        self.sigma_x_edit.setText(f"{self.sigma_x.value() / 10.0:.2f}")
        self.sigma_y_edit.setText(f"{self.sigma_y.value() / 10.0:.2f}")
        self.quant_edit.setText(str(self.quant_slider.value()))
        self.params.sigma_x = self.sigma_x.value() / 10.0
        self.params.sigma_y = self.sigma_y.value() / 10.0
        self.params.edge_algo = self.edge_combo.currentText()
        # Read dynamic edge params
        if self.params.edge_algo == "Canny":
            self.params.canny_low = self.canny_low.value()
            self.params.canny_high = self.canny_high.value()
        elif self.params.edge_algo in ("Sobel", "Laplacian"):
            # For Sobel/Laplacian, set kernel size from combo
            self.params.edge_kernel_size = int(self.edge_kernel_combo.currentData())
        self.params.quant_threshold = self.quant_slider.value()
        # Overlay params
        self.params.overlay_use_zero = self.overlay_zero_radio.isChecked()
        try:
            self.params.pixel_resolution_nm = float(self.res_input.text())
        except Exception:
            self.params.pixel_resolution_nm = 1.0
        self._run_pipeline()
        self._update_ui()

    def _on_sigma_x_edit(self):
        try:
            val = float(self.sigma_x_edit.text())
        except ValueError:
            return
        val = max(0.0, min(5.0, val))
        self.sigma_x.setValue(int(round(val * 10)))
        self.on_params_changed()

    def _on_sigma_y_edit(self):
        try:
            val = float(self.sigma_y_edit.text())
        except ValueError:
            return
        val = max(0.0, min(5.0, val))
        self.sigma_y.setValue(int(round(val * 10)))
        self.on_params_changed()

    def _on_quant_edit(self):
        try:
            val = int(self.quant_edit.text())
        except ValueError:
            return
        val = max(0, min(255, val))
        self.quant_slider.setValue(val)
        self.on_params_changed()

    def _init_edge_param_controls(self):
        # Common controls that may be used depending on algorithm
        # Canny thresholds
        self.canny_low = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.canny_low.setRange(0, 255)
        self.canny_low.setValue(self.params.canny_low)
        self.canny_low.valueChanged.connect(self.on_params_changed)
        self.canny_high = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.canny_high.setRange(0, 255)
        self.canny_high.setValue(self.params.canny_high)
        self.canny_high.valueChanged.connect(self.on_params_changed)
        # Kernel size for Sobel/Laplacian (odd sizes 1,3,5,7)
        self.edge_kernel_combo = QtWidgets.QComboBox()
        for k in [1, 3, 5, 7]:
            self.edge_kernel_combo.addItem(f"{k}", k)
        self.edge_kernel_combo.setCurrentIndex(1)
        self.edge_kernel_combo.currentIndexChanged.connect(self.on_params_changed)

    def _on_edge_algo_changed(self):
        self._rebuild_edge_params()
        self.on_params_changed()

    def _rebuild_edge_params(self):
        # Clear current params layout
        while self.edge_params_layout.count():
            item = self.edge_params_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        algo = self.edge_combo.currentText()
        # Build controls per algorithm
        if algo == "Canny":
            self.edge_params_layout.addRow("Canny Low", self.canny_low)
            self.edge_params_layout.addRow("Canny High", self.canny_high)
        elif algo == "Sobel":
            self.edge_params_layout.addRow("Kernel Size (odd)", self.edge_kernel_combo)
        elif algo == "Laplacian":
            self.edge_params_layout.addRow("Kernel Size (odd)", self.edge_kernel_combo)
        elif algo == "Prewitt":
            lbl = QtWidgets.QLabel("No parameters")
            self.edge_params_layout.addRow(lbl)
        elif algo == "Roberts":
            lbl = QtWidgets.QLabel("No parameters")
            self.edge_params_layout.addRow(lbl)
        elif algo == "Scharr":
            lbl = QtWidgets.QLabel("Fixed operator (no kernel size)")
            self.edge_params_layout.addRow(lbl)

    def _choose_overlay_color(self):
        # QColorDialog uses RGB, convert to BGR for OpenCV
        current_bgr = getattr(self.params, 'overlay_color', (0, 0, 255))
        current_rgb = QtGui.QColor(current_bgr[2], current_bgr[1], current_bgr[0])
        color = QtWidgets.QColorDialog.getColor(current_rgb, self, "Select Overlay Color")
        if color.isValid():
            r, g, b, _ = color.getRgb()
            self.params.overlay_color = (b, g, r)
            self._update_overlay_swatch()
            self._run_pipeline()
            self._update_ui()

    def _update_overlay_swatch(self):
        bgr = getattr(self.params, 'overlay_color', (0, 0, 255))
        r, g, b = bgr[2], bgr[1], bgr[0]
        self.overlay_color_swatch.setStyleSheet(f"background-color: rgb({r}, {g}, {b}); border: 1px solid #666;")

    def _run_pipeline(self):
        if self.input_bgr is None:
            self.results = None
            return
        self.results = process_image(self.input_bgr, self.params)

    def _update_ui(self):
        if self.input_bgr is not None and self.results is not None:
            input_color, eq_gray, smooth_gray, edge_gray, quant_mask, final_overlay, stats = self.results

            self.p_input.set_image(make_pixmap_from_cv(input_color))
            if self.image_path:
                h, w, _ = self.input_bgr.shape
                info = f"File: {os.path.basename(self.image_path)} | Size: {w} x {h}"
            else:
                info = "No image loaded"
            self.input_info.setText(info)

            self.p_eq.set_image(make_pixmap_from_cv(eq_gray))

            self.p_smooth.set_image(make_pixmap_from_cv(smooth_gray))
            self.p_edge.set_image(make_pixmap_from_cv(edge_gray))
            self.p_quant.set_image(make_pixmap_from_cv(quant_mask))
            self.p_final.set_image(make_pixmap_from_cv(final_overlay))

            total_pixels = stats["total_pixels"]
            count_255 = stats.get("count_255", 0)
            percent_255 = stats.get("percent_255", (count_255 / total_pixels * 100.0) if total_pixels else 0.0)
            count_0 = stats.get("count_0", total_pixels - count_255)
            percent_0 = stats.get("percent_0", (count_0 / total_pixels * 100.0) if total_pixels else 0.0)

            overlay_zeros = self.params.overlay_use_zero
            selected_label = "Quantized 0" if overlay_zeros else "Quantized 255"
            selected_count = count_0 if overlay_zeros else count_255
            selected_percent = percent_0 if overlay_zeros else percent_255
            # Edge length in µm: number of edge pixels times pixel resolution (µm)
            edge_length_um = selected_count * self.params.pixel_resolution_nm
            stats_text = (
                f"{selected_label}: {selected_count:,} | %: {selected_percent:.2f}% | "
                f"Edge length: {edge_length_um:,.1f} µm"
            )
            self.stats_label.setText(stats_text)
        else:
            for p in [self.p_input, self.p_eq, self.p_smooth, self.p_edge, self.p_quant, self.p_final]:
                p.set_image(QtGui.QPixmap())
            self.input_info.setText("No image loaded")
            self.stats_label.setText("Stats: N/A")


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = BijelsEdgesApp()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
