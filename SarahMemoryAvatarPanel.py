#!/usr/bin/env python3
"""
SarahMemoryAvatarPanel <Version #7.0 CLI Enhanced>
Author: Brian Lee Baros
This file is strictly for launching the AvatarPanel from SarahMemoryGUI.py

Enhancements:
🖼️ Icon Support
📏 Resizable Toggle (configurable with CLI override)
🔁 Fullscreen Toggle via F11
💾 Graceful Shutdown Hook
🧪 CLI Flags:
    --fullscreen → start in fullscreen
    --fixed      → force fixed-size window
    --resizable  → force resizable window
    --hide-icon  → disable loading the window icon
"""
import sys
import os
import SarahMemoryGlobals as config
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from SarahMemoryGUI import AvatarPanel
from SarahMemoryGlobals import AVATAR_WINDOW_RESIZE, BASE_DIR

class AvatarWindow(QMainWindow):
    def __init__(self, fullscreen=False, force_fixed=False, force_resizable=False, hide_icon=False):
        super().__init__()
        self.setWindowTitle("Avatar Panel GPU Viewer")

        if not hide_icon:
            icon_path = os.path.join(BASE_DIR, "resources", "icons", "avatar_icon.png")
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))

        if force_fixed:
            self.setFixedSize(640, 800)
        elif force_resizable:
            self.setGeometry(300, 100, 640, 800)
        else:
            if not AVATAR_WINDOW_RESIZE:
                self.setFixedSize(640, 800)
            else:
                self.setGeometry(300, 100, 640, 800)

        self.avatar_panel = AvatarPanel()
        self.setCentralWidget(self.avatar_panel)
        self.fullscreen = False

        if fullscreen:
            self.showFullScreen()
            self.fullscreen = True

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F11:
            self.toggle_fullscreen()

    def toggle_fullscreen(self):
        if self.fullscreen:
            self.showNormal()
            self.fullscreen = False
        else:
            self.showFullScreen()
            self.fullscreen = True

    def closeEvent(self, event):
        print("Avatar Panel closed.")
        event.accept()

def main():
    app = QApplication(sys.argv)
    args = sys.argv

    fullscreen = '--fullscreen' in args
    force_fixed = '--fixed' in args
    force_resizable = '--resizable' in args
    hide_icon = '--hide-icon' in args

    window = AvatarWindow(
        fullscreen=fullscreen,
        force_fixed=force_fixed,
        force_resizable=force_resizable,
        hide_icon=hide_icon
    )
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
