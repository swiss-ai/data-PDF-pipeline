import torch
import torch.nn as nn
import torchvision.models as models


class CustomMobileNetV3(nn.Module):
    def __init__(self, num_classes):  # Pass in the actual number of classes
        super(CustomMobileNetV3, self).__init__()
        # Load the pretrained MobileNetV3 model
        self.mobilenet = models.mobilenet_v3_large(pretrained=True)
        
        # Remove the last classifier layer
        self.features = self.mobilenet.features
        
        # Number of output channels from the last conv layer
        # It's important to check the actual number of output channels after the last conv layer
        in_features = self.features[-1].out_channels
        
        # Score fully connected layer, one score for each patch
        self.score_fc = nn.Linear(in_features, 1, bias=False)
        
        # Classifier to output the final class predictions
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # Extract features from the MobileNetV3 backbone
        features = self.features(x)  # Features shape: (batch_size, channels, height, width)
        
        # Flatten the features for each patch
        # The shape becomes (batch_size * height * width, channels)
        # This way we treat each patch as a separate sample for scoring
        batch_size, channels, height, width = features.size()
        flat_features = features.view(batch_size * height * width, channels)
        
        # Obtain a score for each patch
        scores = self.score_fc(flat_features).view(batch_size, 1, height*width)
        # scores = self.score_fc(flat_features)

        # Apply the softmax function to the scores to get weights that sum to 1 for each patch
        
        weights = torch.softmax(scores, dim=2)

        weights = weights.view(batch_size, 1, height, width)
        
        # Multiply each patch in the feature map by its corresponding weight
        weighted_features = features * weights
        
        # Sum the weighted features across the spatial dimensions (height and width)
        # to get a single vector per image
        pooled_features = weighted_features.sum(dim=[2, 3])

        # print(pooled_features)
        
        # Pass the pooled features through the classifier to get final predictions
        out = self.classifier(pooled_features)
        return out