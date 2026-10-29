"""interactive cpu-friendly image gan app"""
from __future__ import annotations  #keep type hints simple

import csv  #save losses so they can be graphed later
import math
import os
import time
from datetime import datetime
from pathlib import Path

try:
    import numpy as np  #work with image pixels
    import torch  #run and save the model
    from PIL import Image, ImageOps  #load and resize images
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    print(f"Missing dependency: {exc}")
    print("Install dependencies using the steps in README.md")
    raise SystemExit(1)

from procedural_dataset import create_interactive as create_procedural_dataset
from tiny_dcgan import BASE_CHANNELS, IMAGE_SIZE, LATENT_DIM, Discriminator, Generator

#keep paths tied to this file
ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "data" / "images"
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "latest.pt"
OUTPUT_DIR = ROOT / "outputs" / "generated"
LOSS_PATH = ROOT / "outputs" / "losses.csv"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}  #image types we can open


def now_tag() -> str:
    #make a unique tag for generated files
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class ImageFolderDataset(Dataset):
    """load local images without torchvision"""

    def __init__(self, folder: Path):
        #search subfolders too, since generated sets may be nested
        self.paths = sorted(
            path for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.paths:
            raise ValueError(
                f"No supported images found in {folder}. Add JPG/PNG/WebP images and try again."
            )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        #load only one source image at a time to save ram
        path = self.paths[index]
        try:
            with Image.open(path) as source:
                image = ImageOps.fit(
                    source.convert("RGB"),
                    (IMAGE_SIZE, IMAGE_SIZE),
                    method=Image.Resampling.LANCZOS,
                    centering=(0.5, 0.5),
                )
                array = np.asarray(image, dtype=np.float32).copy()
        except Exception as exc:
            raise RuntimeError(f"Could not read image {path}: {exc}") from exc

        #convert pixels to channel height width order and scale to the model range
        array = array.transpose(2, 0, 1) / 127.5 - 1.0
        return torch.from_numpy(array)


def save_grid(tensor: torch.Tensor, destination: Path, columns: int = 8) -> None:
    """save images as one contact sheet"""
    if tensor.ndim != 4 or tensor.shape[1] != 3:
        raise ValueError("Expected an NCHW tensor with 3 color channels")

    #convert model output to normal 8-bit pixels
    images = ((tensor.detach().cpu().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8).numpy()
    count = len(images)
    if count == 0:
        return

    columns = max(1, min(columns, count))
    rows = math.ceil(count / columns)
    canvas = Image.new("RGB", (columns * IMAGE_SIZE, rows * IMAGE_SIZE), (24, 24, 24))
    for index, array in enumerate(images):
        image = Image.fromarray(array.transpose(1, 2, 0), mode="RGB")
        x = (index % columns) * IMAGE_SIZE
        y = (index // columns) * IMAGE_SIZE
        canvas.paste(image, (x, y))

    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=95)


class ImageTrainer:
    def __init__(self) -> None:
        #limit cpu threads to keep the pi responsive
        torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
        self.device = torch.device("cpu")
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.loss_function = nn.BCEWithLogitsLoss()
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.epoch = 0
        self.next_batch = 0
        self.step = 0
        self.last_g_loss = float("nan")
        self.last_d_loss = float("nan")
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.load_checkpoint(quiet=True)

    @property
    def has_checkpoint(self) -> bool:
        return CHECKPOINT_PATH.exists()

    def save_checkpoint(self, quiet: bool = False) -> None:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        #keep optimizer state so resumed training continues properly
        payload = {
            "format_version": 2,
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "g_optimizer": self.g_optimizer.state_dict(),
            "d_optimizer": self.d_optimizer.state_dict(),
            "epoch": self.epoch,
            "next_batch": self.next_batch,
            "step": self.step,
            "last_g_loss": self.last_g_loss,
            "last_d_loss": self.last_d_loss,
            "config": {
                "latent_dim": LATENT_DIM,
                "image_size": IMAGE_SIZE,
                "base_channels": BASE_CHANNELS,
            },
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        #replace the old checkpoint only after the new file is saved
        temporary_path = CHECKPOINT_PATH.with_suffix(".tmp")
        torch.save(payload, temporary_path)
        temporary_path.replace(CHECKPOINT_PATH)
        if not quiet:
            print(f"Checkpoint saved: {CHECKPOINT_PATH}")

    def load_checkpoint(self, quiet: bool = False) -> bool:
        #no saved weights means the model starts fresh
        if not CHECKPOINT_PATH.exists():
            if not quiet:
                print("No checkpoint found. Training will start from random weights.")
            return False

        try:
            try:
                #use the safer loader when this torch version supports it
                state = torch.load(CHECKPOINT_PATH, map_location=self.device, weights_only=True)
            except TypeError:
                state = torch.load(CHECKPOINT_PATH, map_location=self.device)

            config = state.get("config", {})
            if config.get("image_size") != IMAGE_SIZE or int(state.get("format_version", 0)) < 2:
                archived = CHECKPOINT_DIR / f"legacy_incompatible_{now_tag()}.pt"
                CHECKPOINT_PATH.replace(archived)
                print(
                    f"Archived an older or incompatible checkpoint to {archived.name}. "
                    f"This model uses {IMAGE_SIZE}x{IMAGE_SIZE} images, so training will start with fresh weights."
                )
                return False

            #restore both networks and their optimizer history
            self.generator.load_state_dict(state["generator"])
            self.discriminator.load_state_dict(state["discriminator"])
            self.g_optimizer.load_state_dict(state["g_optimizer"])
            self.d_optimizer.load_state_dict(state["d_optimizer"])
            self.epoch = int(state.get("epoch", 0))
            self.next_batch = int(state.get("next_batch", 0))
            self.step = int(state.get("step", 0))
            self.last_g_loss = float(state.get("last_g_loss", float("nan")))
            self.last_d_loss = float(state.get("last_d_loss", float("nan")))
            if not quiet:
                print(f"Loaded checkpoint: step {self.step:,}, epoch {self.epoch + 1}")
            return True
        except Exception as exc:
            print(f"Could not load checkpoint ({exc}). The existing file was not deleted.")
            return False

    def reset(self) -> None:
        #reset weights and optimizer history, not just the counters
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.epoch = self.next_batch = self.step = 0
        self.last_g_loss = self.last_d_loss = float("nan")
        if CHECKPOINT_PATH.exists():
            CHECKPOINT_PATH.unlink()
        print("Model reset. Previous checkpoint deleted.")

    def show_status(self) -> None:
        #show a quick text summary for terminal use
        dataset_count = sum(
            1 for path in IMAGE_DIR.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ) if IMAGE_DIR.exists() else 0
        g_params = sum(param.numel() for param in self.generator.parameters())
        d_params = sum(param.numel() for param in self.discriminator.parameters())
        print("\nModel status")
        print(f"  Device: CPU ({torch.get_num_threads()} PyTorch threads)")
        print(f"  Output size: {IMAGE_SIZE} x {IMAGE_SIZE} RGB")
        print(f"  Generator parameters: {g_params:,}")
        print(f"  Discriminator parameters: {d_params:,}")
        print(f"  Training images found: {dataset_count:,}")
        print(f"  Training steps completed: {self.step:,}")
        print(f"  Epoch position: {self.epoch + 1}, next batch {self.next_batch}")
        print(f"  Checkpoint: {'present' if self.has_checkpoint else 'not saved yet'}")
        if self.step == 0:
            print("  Note: generated images will be random until training has progressed.")
        elif math.isfinite(self.last_g_loss) and math.isfinite(self.last_d_loss):
            print(f"  Last losses: generator {self.last_g_loss:.4f}, discriminator {self.last_d_loss:.4f}")
        print()

    def _append_losses(self) -> None:
        #store one loss row at a time to keep the log small
        LOSS_PATH.parent.mkdir(parents=True, exist_ok=True)
        new_file = not LOSS_PATH.exists()
        with LOSS_PATH.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if new_file:
                writer.writerow(["time", "step", "epoch", "generator_loss", "discriminator_loss"])
            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                self.step,
                self.epoch + 1,
                f"{self.last_g_loss:.6f}",
                f"{self.last_d_loss:.6f}",
            ])

    def _train_batch(self, real: torch.Tensor) -> None:
        #train the discriminator on real images and generated fakes
        batch_size = real.shape[0]
        real = real.to(self.device)
        real_targets = torch.full((batch_size,), 0.9, dtype=torch.float32, device=self.device)
        fake_targets = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        generator_targets = torch.ones(batch_size, dtype=torch.float32, device=self.device)

        self.discriminator.zero_grad(set_to_none=True)
        real_logits = self.discriminator(real)
        loss_real = self.loss_function(real_logits, real_targets)
        noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=self.device)
        fake = self.generator(noise).detach()
        loss_fake = self.loss_function(self.discriminator(fake), fake_targets)
        d_loss = loss_real + loss_fake
        d_loss.backward()
        self.d_optimizer.step()

        #train the generator to make fake images look real to the discriminator
        self.generator.zero_grad(set_to_none=True)
        noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=self.device)
        generated = self.generator(noise)
        g_loss = self.loss_function(self.discriminator(generated), generator_targets)
        g_loss.backward()
        self.g_optimizer.step()

        self.step += 1
        self.last_g_loss = float(g_loss.item())
        self.last_d_loss = float(d_loss.item())

    @torch.no_grad()  #sampling images does not need gradients
    def generate(self, count: int = 64) -> Path:
        count = max(1, min(int(count), 256))
        chunks = []
        was_training = self.generator.training
        self.generator.eval()
        remaining = count
        while remaining > 0:
            batch = min(remaining, 32)
            noise = torch.randn(batch, LATENT_DIM, 1, 1, device=self.device)
            chunks.append(self.generator(noise).cpu())
            remaining -= batch

        images = torch.cat(chunks, dim=0)
        tag = now_tag()
        grid_path = OUTPUT_DIR / f"grid_step_{self.step:08d}_{tag}.png"
        save_grid(images, grid_path)
        #save individual files as well as the overview grid
        for index, image in enumerate(images):
            save_grid(image.unsqueeze(0), OUTPUT_DIR / f"sample_{tag}_{index + 1:03d}.png", columns=1)
        self.generator.train(was_training)
        print(f"Saved {count} images. Grid: {grid_path}")
        if self.step == 0:
            print("Warning: the model has not been trained, so these images are random noise-like output.")
        return grid_path

    def train_for_hours(self, hours: float, checkpoint_batches: int = 50, batch_size: int = 4) -> None:
        #load images as needed instead of keeping the full dataset in ram
        try:
            dataset = ImageFolderDataset(IMAGE_DIR)
        except (ValueError, OSError) as exc:
            print(f"Dataset error: {exc}")
            return

        if len(dataset) < 20:
            print(f"Only {len(dataset)} image(s) found. Training can run, but results may not generalize.")
        if len(dataset) < 2:
            print("Add at least two readable images before training.")
            return

        batch_size = max(2, min(32, batch_size))
        checkpoint_batches = max(1, checkpoint_batches)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
        duration_seconds = max(60.0, min(hours * 3600.0, 24 * 30 * 3600.0))
        deadline = time.monotonic() + duration_seconds  #avoid clock changes affecting the timer
        self.generator.train()
        self.discriminator.train()
        start_step = self.step
        start_time = time.monotonic()
        print(f"Training on {len(dataset):,} images, batch size {batch_size}, CPU only")
        print(f"Time budget: {duration_seconds / 3600.0:.2f} hour(s). Checkpoint every {checkpoint_batches} batches")
        print("Press Ctrl+C to save and stop cleanly. The checkpoint is resumed next time.")
        batch_counter = 0

        try:
            while time.monotonic() < deadline:
                epoch_index = self.epoch
                skip_before = self.next_batch
                made_progress = False
                for batch_index, real in enumerate(loader):
                    #skip batches already completed in the saved epoch
                    if epoch_index == self.epoch and batch_index < skip_before:
                        continue
                    if time.monotonic() >= deadline:
                        break

                    self._train_batch(real)
                    batch_counter += 1
                    made_progress = True
                    self.epoch = epoch_index
                    self.next_batch = batch_index + 1

                    if self.step % 20 == 0:
                        elapsed = max(1.0, time.monotonic() - start_time)
                        steps_min = (self.step - start_step) / elapsed * 60
                        print(
                            f"step {self.step:>8,} | epoch {self.epoch + 1:>5,} "
                            f"| D {self.last_d_loss:.4f} | G {self.last_g_loss:.4f} "
                            f"| {steps_min:.2f} steps/min",
                            flush=True,
                        )

                    if batch_counter % checkpoint_batches == 0:
                        self.save_checkpoint(quiet=True)
                        self._append_losses()
                    if self.step > 0 and self.step % 250 == 0:
                        self.generate(16)
                        self.generator.train()
                    if time.monotonic() >= deadline:
                        break

                if not made_progress and time.monotonic() >= deadline:
                    break
                if self.next_batch >= len(loader):
                    self.epoch = epoch_index + 1
                    self.next_batch = 0
                    self.save_checkpoint(quiet=True)
                    print(f"Completed epoch {self.epoch:,}", flush=True)
                    self.generate(16)
                    self.generator.train()
                elif not made_progress:
                    break
        except KeyboardInterrupt:
            print("\nStop requested. Saving current progress...")
        except Exception as exc:
            print(f"\nTraining stopped because of an error: {type(exc).__name__}: {exc}")
            raise
        finally:
            #save the latest state on exit so training can resume
            self.save_checkpoint(quiet=True)
            elapsed = max(0.001, time.monotonic() - start_time)
            trained = self.step - start_step
            if trained > 0:
                self._append_losses()
            print(
                f"Saved checkpoint at step {self.step:,}. This session trained "
                f"{trained:,} steps in {elapsed / 3600:.2f} hour(s)."
            )


def ask_int(prompt: str, default: int, minimum: int, maximum: int) -> int:
    #keep asking until a valid whole number is entered
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = int(raw)
            if minimum <= value <= maximum:
                return value
        except ValueError:
            pass
        print(f"Enter a whole number from {minimum} to {maximum}")


def ask_float(prompt: str, default: float, minimum: float, maximum: float) -> float:
    #allow decimal values such as training hours
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = float(raw)
            if minimum <= value <= maximum:
                return value
        except ValueError:
            pass
        print(f"Enter a number from {minimum} to {maximum}")


def show_menu() -> None:
    print("\n======================================")
    print(" Raspberry Pi Tiny Image Lab — 64x64")
    print("======================================")
    print("1. Check image dataset")
    print("2. Create procedural training images")
    print("3. Train or resume training")
    print("4. Generate images")
    print("5. Show model and checkpoint status")
    print("6. Reset model and start over")
    print("7. Save checkpoint now")
    print("8. Quit")


def check_dataset() -> None:
    #test that at least one image can be opened and resized
    try:
        dataset = ImageFolderDataset(IMAGE_DIR)
    except ValueError as exc:
        print(exc)
        return

    print(f"Found {len(dataset):,} image files in {IMAGE_DIR}")
    print(f"The loader center-crops and resizes them to {IMAGE_SIZE}x{IMAGE_SIZE} RGB")
    print("Start with 200–500 similar images; larger, relevant sets can add variety")
    print("Use images you have permission to train on. Variety and consistency matter")
    try:
        sample = dataset[0]
        print(f"First image loads successfully: tensor shape {tuple(sample.shape)}")
    except Exception as exc:
        print(f"The first image did not load: {exc}")


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer = ImageTrainer()
    print("64x64 RGB image generator | PyTorch CPU mode")
    if trainer.has_checkpoint:
        print(f"Checkpoint found. Resumed step {trainer.step:,} automatically.")
    else:
        print("No 64x64 checkpoint yet. Add or create training images, then choose option 3.")

    while True:
        show_menu()
        choice = input("Select a number: ").strip()
        if choice == "1":
            check_dataset()
        elif choice == "2":
            create_procedural_dataset()
        elif choice == "3":
            hours = ask_float("Training duration in hours (168 = one week)", 168.0, 0.02, 720.0)
            batch_size = ask_int("Batch size (smaller uses less RAM)", 4, 2, 16)
            checkpoint_batches = ask_int("Save checkpoint every N batches", 25, 1, 1000)
            trainer.train_for_hours(hours, checkpoint_batches, batch_size)
        elif choice == "4":
            count = ask_int("How many images to generate", 64, 1, 256)
            trainer.generate(count)
        elif choice == "5":
            trainer.show_status()
        elif choice == "6":
            confirm = input("This deletes learned weights. Type RESET to confirm: ").strip()
            if confirm == "RESET":
                trainer.reset()
            else:
                print("Reset cancelled")
        elif choice == "7":
            trainer.save_checkpoint()
        elif choice == "8":
            trainer.save_checkpoint(quiet=True)
            print("Checkpoint saved. Goodbye.")
            return
        else:
            print("Choose one of the listed numbers (1-8)")


if __name__ == "__main__":
    try:
        main()
    except (EOFError, BrokenPipeError):
        print("Input closed. Exiting")
