import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from gan_modules import Generator, Discriminator


def generate_images():
    # --------------------
    # Generate image samples from different Fashion-MNIST classes
    # --------------------
    generator.eval()

    samples_per_class = 5
    num_classes = len(dataset.classes)

    sample_labels = torch.arange(num_classes).repeat_interleave(samples_per_class).to(device)

    num_samples = sample_labels.size(0)

    z = torch.randn(num_samples, latent_dim).to(device)

    with torch.no_grad():
        generated_imgs = generator(z, sample_labels)

    generated_imgs = generated_imgs.cpu()
    sample_labels = sample_labels.cpu()

    # Denormalize from [-1, 1] to [0, 1]
    generated_imgs = (generated_imgs + 1) / 2
    generated_imgs = generated_imgs.clamp(0, 1)

    fig, axes = plt.subplots(
        num_classes,
        samples_per_class,
        figsize=(samples_per_class * 2, num_classes * 2)
    )

    for class_idx in range(num_classes):
        for sample_idx in range(samples_per_class):
            img_idx = class_idx * samples_per_class + sample_idx

            axes[class_idx, sample_idx].imshow(
                generated_imgs[img_idx].squeeze(),
                cmap="gray"
            )

            axes[class_idx, sample_idx].axis("off")

            if sample_idx == 0:
                axes[class_idx, sample_idx].set_ylabel(
                    dataset.classes[class_idx],
                    fontsize=10
                )

    plt.suptitle("Generated Samples from Different Fashion-MNIST Classes", fontsize=16)
    plt.tight_layout()
    plt.savefig("GAN/generated_samples_grid.png", dpi=300)
    plt.show()


def compare_saved_stages(device, num_classes, latent_dim):

    stage_files = [
        ("Early", "GAN/gen_early.pth"),
        ("Mid", "GAN/gen_mid.pth"),
        ("Final", "GAN/gen_final.pth")
    ]

    fixed_noise = torch.randn(num_classes, latent_dim).to(device)
    fixed_labels = torch.arange(num_classes).to(device)

    fig, axes = plt.subplots(len(stage_files), num_classes, figsize=(15, 8))

    for row_idx, (stage_name, model_file) in enumerate(stage_files):

        # load generator
        temp_gen = Generator(
            latent_dim=latent_dim,
            num_classes=num_classes
        ).to(device)

        temp_gen.load_state_dict(torch.load(model_file, map_location=device))

        temp_gen.eval()

        with torch.no_grad():
            generated_imgs = temp_gen(fixed_noise, fixed_labels)

        generated_imgs = generated_imgs.cpu()

        # denormalize
        generated_imgs = (generated_imgs + 1) / 2
        generated_imgs = generated_imgs.clamp(0, 1)

        for col_idx in range(num_classes):

            axes[row_idx, col_idx].imshow(
                generated_imgs[col_idx].squeeze(),
                cmap='gray'
            )

            axes[row_idx, col_idx].axis("off")

            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel(
                    stage_name,
                    fontsize=14
                )

    plt.suptitle(
        "Comparison of Generated Images at Different Training Stages",
        fontsize=16
    )

    plt.tight_layout()

    plt.savefig("GAN/training_stage_comparison.png", dpi=300)

    plt.show()


if __name__ == '__main__':

    # --------------------
    # Hyperparameters
    # --------------------
    batch_size = 64
    latent_dim = 100
    num_epochs =200 ######50
    lr = 0.0002

    # --------------------
    # Device
    # --------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --------------------
    # Loading FashionMNIST
    # --------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = datasets.FashionMNIST(
        root="./data/FashionMNIST",
        train=True,
        download=True,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    num_classes = len(dataset.classes)

    # --------------------
    # Initialize models
    # --------------------
    generator = Generator(latent_dim=latent_dim, num_classes=num_classes).to(device)
    discriminator = Discriminator(num_classes=num_classes).to(device)

    # --------------------
    # Loss and optimizers (Binary cross entropy)
    # --------------------
    adversarial_loss = nn.BCELoss()

    optimizer_G = torch.optim.Adam(
        generator.parameters(),
        lr=lr,
        betas=(0.5, 0.999)
    )

    optimizer_D = torch.optim.Adam(
        discriminator.parameters(),
        lr=lr,
        betas=(0.5, 0.999)
    )
    # --------------------
    # Training loop
    # --------------------
    G_losses = []
    D_losses = []

    for epoch in range(num_epochs):
        for batch_idx, (imgs, labels) in enumerate(dataloader):

            imgs = imgs.to(device)
            labels = labels.to(device).long()

            current_batch_size = imgs.size(0)

            valid = torch.ones(current_batch_size, 1).to(device)
            fake = torch.zeros(current_batch_size, 1).to(device)

            # -------------------
            # Train Generator
            # -------------------
            optimizer_G.zero_grad()

            z = torch.randn(current_batch_size, latent_dim).to(device)

            gen_imgs = generator(z, labels)

            g_loss = adversarial_loss(
                discriminator(gen_imgs, labels),
                valid
            )

            g_loss.backward()
            optimizer_G.step()

            # -------------------
            # Train Discriminator
            # -------------------
            optimizer_D.zero_grad()

            real_loss = adversarial_loss(
                discriminator(imgs, labels),
                valid
            )

            fake_loss = adversarial_loss(
                discriminator(gen_imgs.detach(), labels),
                fake
            )

            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"D Loss: {d_loss.item():.4f} | "
            f"G Loss: {g_loss.item():.4f}"
        )

        # --- different stages weights ---
        if epoch == 0:
            torch.save(generator.state_dict(), "GAN/gen_early.pth")
        elif epoch == num_epochs // 2:
            torch.save(generator.state_dict(), "GAN/gen_mid.pth")
        elif epoch == num_epochs - 1:
            torch.save(generator.state_dict(), "GAN/gen_final.pth")

    print("Training finished. Starting stage comparison...")
    compare_saved_stages(device, num_classes, latent_dim)



    # --------------------
    # Plot Losses
    # --------------------
    plt.figure(figsize=(10, 5))
    plt.plot(G_losses, label="Generator")
    plt.plot(D_losses, label="Discriminator")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Generator and Discriminator Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("GAN/loss_curves.png")
    plt.show()

    # -------------------
    # Generate images
    # -------------------
    generate_images()

