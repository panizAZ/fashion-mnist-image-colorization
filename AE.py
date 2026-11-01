import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import os
from AE_modules import ColorizationDataset, ColorizationAutoencoder128
from skimage.metrics import peak_signal_noise_ratio as psnr_metric, structural_similarity as ssim_metric, mean_squared_error as mse_metric

if __name__ == "__main__":
    # initialize
    RESUME_TRAINING = False  # If True, continues training a pretrained saved model
    MODEL_PATH = "AE_files/colorization_model.pth"
    num_epochs = 200
    batch_size = 16
    lr = 0.0001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)


    full_dataset = ColorizationDataset(root_dir="./data/AE Data/train", img_size=128)

    # train and validation data
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    test_dataset = ColorizationDataset(root_dir="./data/AE Data/test", img_size=128)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = ColorizationAutoencoder128().to(device)

    # for initializing weights
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    model.apply(init_weights)

    if RESUME_TRAINING and os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print(f"مدل از فایل {MODEL_PATH} لود شد و آموزش ادامه می‌یابد.")
    else:
        print("آموزش از ابتدا شروع می‌شود.")

    #criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    # Early Stopping
    patience = 8  # epochs it waits for early stopping
    patience_counter = 0
    best_val_loss = float('inf')
    best_model_state = None

    # Training loop
    print("Starting Training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for gray, rgb in train_loader:
            gray, rgb = gray.to(device), rgb.to(device)

            outputs = model(gray)
            loss = criterion(outputs, rgb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation loop
        model.eval()
        val_running = 0
        with torch.no_grad():
            for gray, rgb in val_loader:
                gray, rgb = gray.to(device), rgb.to(device)
                preds = model(gray)
                val_running += criterion(preds, rgb).item()

        avg_val_loss = val_running / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        # Early Stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # ذخیره کردن بهترین وزن‌های مدل تا این لحظه در حافظه موقت
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            print(f"Early Stopping Counter: {patience_counter} out of {patience}")
            if patience_counter >= patience:
                print("Early stopping triggered. Training stopped.")
                # بارگذاری بهترین وزن‌ها روی مدل
                model.load_state_dict(best_model_state)
                break

    # Saving Model
    os.makedirs("AE_files", exist_ok=True)
    torch.save(model.state_dict(), "AE_files/colorization_model.pth")

    # loss plot
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.title("Loss Curves")
    plt.legend()
    plt.savefig("AE_files/loss_curves.png")

    # test and metrics calculation
    model.eval()
    total_psnr, total_ssim, total_mse,count = 0, 0, 0, 0
    os.makedirs("AE_files/all_outputs", exist_ok=True)

    with torch.no_grad():
        for i, (gray, rgb) in enumerate(test_loader):
            gray, rgb = gray.to(device), rgb.to(device)
            preds = model(gray)

            # transformation for metrics calculations and saving images
            rgb_np = rgb.permute(0, 2, 3, 1).cpu().numpy()
            pred_np = preds.permute(0, 2, 3, 1).cpu().numpy()
            gray_np = gray.squeeze(1).cpu().numpy()

            for j in range(len(rgb_np)):
                #  PSNR & SSIM calculations
                total_psnr += psnr_metric(rgb_np[j], pred_np[j], data_range=1.0)
                total_ssim += ssim_metric(rgb_np[j], pred_np[j], channel_axis=2, data_range=1.0)
                total_mse += mse_metric(rgb_np[j], pred_np[j])

                # Saving samples
                if count < 50:  # if test was more than 10 , it only looks at 50
                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    axes[0].imshow(gray_np[j], cmap='gray');
                    axes[0].set_title("Input")
                    axes[1].imshow(pred_np[j])
                    axes[1].set_title("Output")
                    axes[2].imshow(rgb_np[j])
                    axes[2].set_title("Target")
                    for ax in axes: ax.axis('off')
                    plt.savefig(f"AE_files/all_outputs/result_{count}.png")
                    plt.close()
                count += 1

    print(f"\nFinal Test Results:")
    print(f"PSNR: {total_psnr / count:.2f} dB")
    print(f"SSIM: {total_ssim / count:.4f}")
    print(f"MSE:  {total_mse / count:.6f}")