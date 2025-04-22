import matplotlib.pyplot as plt
import seaborn as sns
from logging import getLogger, INFO, Formatter, StreamHandler

# Logger setup
plot_logger = getLogger("VizUtils")
plot_logger.setLevel(INFO)
handler = StreamHandler()
handler.setFormatter(Formatter("%(asctime)s - [%(levelname)s] - %(message)s"))
plot_logger.addHandler(handler)

def plot_metrics_bar(metrics, model_name, output_file="metrics_bar.png", show=False):
    """
    Plot a bar chart of model performance metrics.

    Args:
        metrics (dict): Dictionary of metrics (e.g., accuracy, precision).
        model_name (str): Name of the model.
        output_file (str): Path to save the plot.
        show (bool): If True, display the plot interactively.
    """
    try:
        plt.figure(figsize=(8, 5))
        sns.barplot(x=list(metrics.values()), y=list(metrics.keys()), palette="coolwarm")
        plt.title(f"Performance Metrics for {model_name}")
        plt.xlabel("Score")
        plt.xlim(0, 1)
        for i, v in enumerate(metrics.values()):
            plt.text(v + 0.01, i, f"{v:.2f}", va="center")
        plt.savefig(output_file)
        plot_logger.info(f"Metrics bar plot for {model_name} saved to {output_file}")
        if show:
            plt.show()
    except Exception as e:
        plot_logger.error(f"Failed to plot metrics bar for {model_name}: {str(e)}")
    finally:
        plt.close()