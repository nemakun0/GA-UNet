"""
Live Training Plot — Eğitim sırasında canlı güncellenen grafikler.
=================================================================
TkAgg backend kullanır, her epoch sonunda grafikleri günceller.
Headless ortamlarda --no-live-plot ile devre dışı bırakılabilir.
"""

import matplotlib
import matplotlib.pyplot as plt


class LivePlotCallback:
    """
    Her epoch sonunda çağrılan canlı grafik güncelleyici.

    Kullanım:
        live = LivePlotCallback()           # __init__ pencereyi açar
        for epoch in ...:
            ...
            live.update(history)            # grafikler güncellenir
        live.close()                        # pencere kapanır
    """

    # ── Renk paleti (koyu tema) ──
    BG    = '#0f1117'
    PANEL = '#1a1d2e'
    GRID  = '#2a2d3e'
    TEXT  = '#e0e0f0'
    BLUE  = '#4C9BE8'
    RED   = '#E8694C'
    GREEN = '#4CE8A0'
    PURPLE = '#B44CE8'

    def __init__(self):
        matplotlib.use('TkAgg')
        plt.ion()

        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 8))
        self.fig.patch.set_facecolor(self.BG)
        self.fig.suptitle('GA-UNet — Canlı Eğitim Takibi',
                          color=self.TEXT, fontsize=14, fontweight='bold')

        self._titles = ['Loss', 'Dice Similarity Coefficient',
                        'Validation IoU', 'Learning Rate']
        for ax, title in zip(self.axes.flat, self._titles):
            self._style(ax, title)

        self.fig.tight_layout(rect=[0, 0, 1, 0.94])
        self.fig.canvas.draw()
        self.fig.show()

    # ── Grafik stili ──
    def _style(self, ax, title):
        ax.set_facecolor(self.PANEL)
        ax.set_title(title, color=self.TEXT, fontsize=10, fontweight='bold')
        ax.tick_params(colors=self.TEXT, labelsize=8)
        ax.xaxis.label.set_color(self.TEXT)
        ax.yaxis.label.set_color(self.TEXT)
        for sp in ax.spines.values():
            sp.set_edgecolor(self.GRID)
        ax.grid(True, color=self.GRID, linewidth=0.6, alpha=0.8)

    # ── Epoch sonunda çağır ──
    def update(self, history: dict):
        """history dict'indeki verileri canlı olarak grafiğe yansıtır."""
        epochs = list(range(1, len(history['train_loss']) + 1))
        if not epochs:
            return

        ax_loss, ax_dsc, ax_iou, ax_lr = self.axes.flat

        # — Loss —
        ax_loss.clear()
        self._style(ax_loss, self._titles[0])
        ax_loss.plot(epochs, history['train_loss'], color=self.BLUE,
                     lw=2, marker='o', ms=3, label='Train')
        ax_loss.plot(epochs, history['val_loss'], color=self.RED,
                     lw=2, marker='s', ms=3, label='Val')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend(fontsize=8, labelcolor=self.TEXT,
                       facecolor=self.PANEL, edgecolor=self.GRID)

        # — DSC —
        ax_dsc.clear()
        self._style(ax_dsc, self._titles[1])
        ax_dsc.plot(epochs, history['train_dsc'], color=self.BLUE,
                    lw=2, marker='o', ms=3, label='Train')
        ax_dsc.plot(epochs, history['val_dsc'], color=self.RED,
                    lw=2, marker='s', ms=3, label='Val')
        ax_dsc.set_xlabel('Epoch')
        ax_dsc.set_ylabel('DSC')
        ax_dsc.legend(fontsize=8, labelcolor=self.TEXT,
                      facecolor=self.PANEL, edgecolor=self.GRID)

        # — IoU —
        ax_iou.clear()
        self._style(ax_iou, self._titles[2])
        ax_iou.plot(epochs, history['val_iou'], color=self.GREEN,
                    lw=2, marker='^', ms=3, label='Val IoU')
        ax_iou.set_xlabel('Epoch')
        ax_iou.set_ylabel('IoU')
        ax_iou.legend(fontsize=8, labelcolor=self.TEXT,
                      facecolor=self.PANEL, edgecolor=self.GRID)

        # — LR —
        ax_lr.clear()
        self._style(ax_lr, self._titles[3])
        ax_lr.plot(epochs, history['lr'], color=self.PURPLE,
                   lw=2, marker='o', ms=2)
        ax_lr.set_xlabel('Epoch')
        ax_lr.set_ylabel('Learning Rate')
        ax_lr.set_yscale('log')

        self.fig.tight_layout(rect=[0, 0, 1, 0.94])
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    # ── Eğitim bittiğinde kapat ──
    def close(self):
        """Live plot penceresini kapatır."""
        plt.ioff()
        plt.close(self.fig)
