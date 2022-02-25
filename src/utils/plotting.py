"""Methods for plotting
"""
from matplotlib import figure
from matplotlib.backends.backend_pdf import PdfPages


class PdfFile:
    """PdfPages object to store figures as pdf pages."""

    def __init__(self, fig_path):
        """Open file.

        Args:
            fig_path (str): Path to store the pdf file.
        """
        self.pdf = PdfPages(fig_path)

    def close(self):
        """Close file."""
        self.pdf.close()

    def save_fig(self, data, title):
        """Visualizes the metric scores

        Args:
            data (pd.DataFrame): The Dataframe contains sales and forecasts data
        """
        # fill missing days as nan to break the lines in the plot
        data.sales = data.sales.asfreq("D")
        data.prediction = data.prediction.asfreq("D")

        fig = figure.Figure(figsize=(8, 8))
        axs = fig.subplots(squeeze=False, sharex=True, sharey=True)
        axs = axs.flat

        axs[0].plot(data.index, data.sales, label="sales", alpha=0.5, c="C0")
        axs[0].plot(data.index, data.prediction, label="prediction", alpha=0.5, c="C1")

        fig.suptitle(title)

        self.pdf.savefig(fig)
        fig.clf()
