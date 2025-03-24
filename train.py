import torch
from tqdm import tqdm

def get_device():
    """Check if a GPU is available."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, train_loader, val_loader, device, num_epochs=10, learning_rate=1e-4):
    """Train the model"""
    model = model.to(device)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # -----------------
        # Training Phase
        # -----------------
        model.train()
        total_train_loss = 0

        for inputs, heatmaps, visibility in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)"):
            inputs = inputs.to(device)
            heatmaps = heatmaps.to(device)
            visibility = visibility.to(device).float()  # Ensure it is a floating-point number

            # Forward propagation
            stage_outputs = model(inputs)

            # Compute loss step by step
            loss = 0
            for stage_out in stage_outputs:
                visibility_mask = visibility.unsqueeze(-1).unsqueeze(-1).expand_as(heatmaps)
                visible_heatmaps = heatmaps * visibility_mask
                visible_stage_out = stage_out * visibility_mask
                visible_pixel_count = visibility_mask.sum()

                if visible_pixel_count > 0:
                    loss += criterion(visible_stage_out, visible_heatmaps) / visible_pixel_count

            # Accumulate to total training loss
            total_train_loss += loss.item()

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print training loss
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {total_train_loss:.4f}")

        # -----------------
        # Validation Phase
        # -----------------
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for inputs, heatmaps, visibility in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Validation)"):
                inputs = inputs.to(device)
                heatmaps = heatmaps.to(device)
                visibility = visibility.to(device).float()

                # Forward propagation
                stage_outputs = model(inputs)

                # Compute val loss step by step
                loss = 0
                for stage_out in stage_outputs:
                    visibility_mask = visibility.unsqueeze(-1).unsqueeze(-1).expand_as(heatmaps)
                    visible_heatmaps = heatmaps * visibility_mask
                    visible_stage_out = stage_out * visibility_mask
                    visible_pixel_count = visibility_mask.sum()

                    if visible_pixel_count > 0:
                        loss += criterion(visible_stage_out, visible_heatmaps) / visible_pixel_count

                # Accumulate to total val loss
                total_val_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {total_val_loss:.4f}")

        # Save the model with the best performance on the validation set
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            torch.save(model.state_dict(), "best_model_weights_val.pth")
            print(f"Saved best model at epoch {epoch+1} with validation loss {best_val_loss:.4f}")
        
        # Save the entire model
        torch.save(model, "final_model.pth")





