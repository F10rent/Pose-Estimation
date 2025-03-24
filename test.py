import torch
import numpy as np
import matplotlib.pyplot as plt

def get_keypoints_from_heatmap(heatmaps):
    """
    Extract keypoint coordinates from heatmaps.
    heatmaps: numpy array, shape = (batch_size, num_joints, height, width)
    Returns: Keypoint coordinates, shape = (batch_size, num_joints, 2)
    """
    batch_size, num_joints, height, width = heatmaps.shape
    keypoints = np.zeros((batch_size, num_joints, 2), dtype=np.float32)

    for i in range(batch_size):
        for j in range(num_joints):
            heatmap = heatmaps[i, j]
            # Get the positions of the maximum values
            idx = np.argmax(heatmap)
            y, x = np.unravel_index(idx, heatmap.shape)
            keypoints[i, j] = [x, y]  # Keypoint coordinates as (x, y)
    return keypoints

def calculate_pck(pred_keypoints, gt_keypoints, image_size, alpha_values):
    """
    Calculate multi-threshold PCK (Percentage of Correct Keypoints).
    pred_keypoints: numpy array, shape = (batch_size, num_joints, 2)
    gt_keypoints: numpy array, shape = (batch_size, num_joints, 2)
    image_size: tuple, image dimensions (height, width)
    alpha_values: list, different alpha thresholds
    Returns: PCK values for each alpha threshold
    """
    pck_results = {}
    batch_size, num_joints, _ = pred_keypoints.shape
    max_dim = max(image_size)  # Use the longer side to calculate the threshold

    for alpha in alpha_values:
        threshold = alpha * max_dim
        correct_count = 0
        total_count = batch_size * num_joints

        for i in range(batch_size):
            for j in range(num_joints):
                # Calculate Euclidean distance
                distance = np.linalg.norm(pred_keypoints[i, j] - gt_keypoints[i, j])
                if distance <= threshold:
                    correct_count += 1

        # Record the PCK for each alpha
        pck_results[alpha] = correct_count / total_count

    return pck_results


def test_model(model, test_loader, device):
    """
    Test the model's performance.

    Parameters:
        model: torch.nn.Module
            The trained model.
        test_loader: DataLoader
            DataLoader for the test dataset.
        device: torch.device
            Device (CPU or GPU).

    Returns:
        float
            Total loss on the test dataset.
    """
    # Switch to evaluation mode
    model.eval()
    total_loss = 0
    criterion = torch.nn.MSELoss(reduction='sum')  # Loss function consistent with training

    # Disable gradient computation
    with torch.no_grad():
        for inputs, heatmaps, visibility in test_loader:
            inputs = inputs.to(device)
            heatmaps = heatmaps.to(device)
            visibility = visibility.to(device).float()

            # Model inference
            stage_outputs = model(inputs)

            # Compute test set loss (step by step)
            loss = 0
            for stage_out in stage_outputs:
                visibility_mask = visibility.unsqueeze(-1).unsqueeze(-1).expand_as(heatmaps)
                visible_heatmaps = heatmaps * visibility_mask
                visible_stage_out = stage_out * visibility_mask
                visible_pixel_count = visibility_mask.sum()

                if visible_pixel_count > 0:
                    loss += criterion(visible_stage_out, visible_heatmaps) / visible_pixel_count

            # Accumulate loss
            total_loss += loss.item()

    print(f"Test Loss: {total_loss:.4f}")
 
def get_predictions_ground_truths_heatmaps(model, test_loader, device): 
    predictions = []  # Used to save predictions
    ground_truths = []  # Used to save ground truth values

    with torch.no_grad():
        for inputs, heatmaps, visibility in test_loader:
            inputs = inputs.to(device)
            heatmaps = heatmaps.to(device)
            visibility = visibility.to(device).float()

            # Model inference
            stage_outputs = model(inputs)
            final_output = stage_outputs[-1]  # The output of the last stage is the final result

            # Save the results
            predictions.append(final_output.cpu().numpy())
            ground_truths.append(heatmaps.cpu().numpy())

    # Convert to NumPy arrays for further analysis
    predictions = np.concatenate(predictions, axis=0)
    ground_truths = np.concatenate(ground_truths, axis=0)

    return predictions, ground_truths, heatmaps

def visualise_heapmat(predictions, ground_truths, number):

    num_joints = 14  # Number of joints

    for i in range(number):  # Visualize the first number of test samples
        plt.figure(figsize=(15, 8))
    
        # Visualize Ground Truth
        for joint_idx in range(num_joints):
            plt.subplot(2, num_joints, joint_idx + 1) # Top part shows Ground Truth
            plt.title(f"GT Joint {joint_idx + 1}")
            plt.imshow(ground_truths[i, joint_idx], cmap="hot")
            plt.axis("off")
    
        # Visualize Predictions
        for joint_idx in range(num_joints):
            plt.subplot(2, num_joints, joint_idx + 1 + num_joints)  # Bottom part shows Prediction
            plt.title(f"Pred Joint {joint_idx + 1}")
            plt.imshow(predictions[i, joint_idx], cmap="hot")
            plt.axis("off")
    
        plt.suptitle(f"Sample {i + 1}")
        plt.tight_layout()
        plt.show()

def get_pck(predictions, ground_truths, heatmaps):
    # Extract keypoint coordinates
    pred_keypoints = get_keypoints_from_heatmap(predictions)  # Extract keypoints from predicted heatmaps
    gt_keypoints = get_keypoints_from_heatmap(ground_truths)  # Extract keypoints from ground truth heatmaps

    # Define input image size and alpha values for PCK
    image_size = (heatmaps.shape[2], heatmaps.shape[3])  # Input image size is consistent with heatmap size
    alpha_values = [0.05, 0.1, 0.2]  # Set the threshold ratio

    # Calculate PCK
    pck_results = calculate_pck(pred_keypoints, gt_keypoints, image_size, alpha_values)

    for alpha, pck in pck_results.items():
        print(f"PCK (alpha={alpha:.2f}): {pck:.4f}")

    # Plot PCK curve
    alphas = list(pck_results.keys())
    pck_values = list(pck_results.values())

    plt.plot(alphas, pck_values, marker='o')
    plt.title("PCK Curve")
    plt.xlabel("Alpha (Threshold Ratio)")
    plt.ylabel("PCK")
    plt.grid(True)
    plt.show()
