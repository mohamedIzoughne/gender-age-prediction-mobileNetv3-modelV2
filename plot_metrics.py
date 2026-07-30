import csv
import matplotlib.pyplot as plt

def plot_metrics(csv_path):
    epochs = []
    train_age = []
    val_age = []
    train_gender = []
    val_gender = []

    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['stage'] == 'train':
                epochs.append(int(row['epoch']))
                train_age.append(float(row['train_age_mae']))
                val_age.append(float(row['val_age_mae']))
                train_gender.append(float(row['train_gender_acc']))
                val_gender.append(float(row['val_gender_acc']))
    
    # Figure 1: Age MAE
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_age, label='Train Age MAE', marker='o')
    plt.plot(epochs, val_age, label='Val Age MAE', marker='o')
    plt.title('Age Mean Absolute Error (MAE) over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    plt.savefig('age_mae_plot.png')
    plt.close()
    
    # Figure 2: Gender Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_gender, label='Train Gender Accuracy', marker='o')
    plt.plot(epochs, val_gender, label='Val Gender Accuracy', marker='o')
    plt.title('Gender Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('gender_acc_plot.png')
    plt.close()
    
    print("Plots saved successfully:")
    print(" - age_mae_plot.png")
    print(" - gender_acc_plot.png")

if __name__ == "__main__":
    plot_metrics('mobilenet_v3_large_aug_metrics.csv')

