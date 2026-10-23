import csv
import matplotlib.pyplot as plt


def plot_compare_training_results(csv_file_path1, csv_file_path2, label1='Dataset 1', label2='Dataset 2'):
    epochs1, losses1, accuracies1 = [], [], []
    epochs2, losses2, accuracies2 = [], [], []

    # 读取第一个CSV文件
    with open(csv_file_path1, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行
        for row in reader:
            epochs1.append(int(row[0]))
            losses1.append(float(row[1]))
            accuracies1.append(float(row[2]))

    # 读取第二个CSV文件
    with open(csv_file_path2, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行
        for row in reader:
            epochs2.append(int(row[0]))
            losses2.append(float(row[1]))
            accuracies2.append(float(row[2]))

    # 绘制训练损失对比
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs1, losses1, label=f'Train Loss ({label1})', color='red')
    plt.plot(epochs2, losses2, label=f'Train Loss ({label2})', color='blue')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, 1)  # 设置y轴的范围从0到1
    plt.grid(True)
    plt.legend()

    # 绘制训练准确率对比
    plt.subplot(1, 2, 2)
    plt.plot(epochs1, accuracies1, label=f'Train Accuracy ({label1})', color='red')
    plt.plot(epochs2, accuracies2, label=f'Train Accuracy ({label2})', color='blue')
    plt.title('Training Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()


    plt.tight_layout()
    plt.show()

# plot_compare_training_results('./results/MUTAG_results.csv', './results/MUTAG_SAGNN_results.csv', label1='Experiment 1', label2='Experiment 2')
plot_compare_training_results('./results/exp_results.csv', './results/exp_SAGNN_results.csv', label1='MPNN', label2='SAGNN')
plot_compare_training_results('./results/CEXP_results.csv', './results/CEXP_SAGNN_results.csv', label1='MPNN', label2='SAGNN')

plot_compare_training_results('./results_1000/ZINC_results.csv', './results_1000/ZINC_SAGNN_results.csv', label1='MPNN', label2='SAGNN')
plot_compare_training_results('./results/ZINC_results.csv', './results/ZINC_SAGNN_results.csv', label1='MPNN', label2='SAGNN')
