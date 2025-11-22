import torch

# from torchvision.io import decode_image
# from torchvision.models import ResNet18_Weights, resnet18
import torch.nn.functional as F


def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on test data.

    Args:
        model: The trained model
        test_loader: DataLoader for test data
        device: Device to evaluate on

    Returns:
        accuracy: Test accuracy
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


def classify_single_image(model, image, device, class_names=None):
    """
    Classify a single image.

    Args:
        model: Trained model
        image: Input image tensor (C, H, W)
        device: Device to run inference on
        class_names: List of class names (optional)

    Returns:
        predicted_class: Predicted class index
        confidence: Prediction confidence
    """
    model.eval()

    # Add batch dimension if needed
    if image.dim() == 3:
        image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = predicted.item()
    confidence_score = confidence.item()

    return predicted_class, confidence_score
