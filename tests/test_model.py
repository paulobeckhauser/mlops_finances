# import torch
# from finance.model import DeepLearningModel

# @pytest.fixture
# def model():
#     return DeepLearningModel(input_size=10, num_classes=3, lr=0.01)

# def test_model_test_step(model):
#     """Test the test step."""
#     # sample_input = torch.rand(5, 10)  # Batch of 5, input size 10
#     # sample_labels = torch.randint(0, 3, (5,))  # Batch of 5, num_classes=3
#     # batch = (sample_input, sample_labels)
#     # loss = model.test_step(batch, batch_idx=0)
#     # assert loss > 0, "Test loss should be greater than 0"