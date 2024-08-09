import deeplake
import matplotlib.pyplot as plt

ds = deeplake.load("hub://activeloop/pacs-test")

print(ds)

sample_index = 9990
sample = ds[sample_index]

image = sample['images'].numpy()
class_label = sample['labels'].numpy()
domain_label = sample['domains'].numpy()

print(f"Class Label: {class_label}")
print(f"Domain Label: {domain_label}")

plt.imshow(image)
plt.title(f"Class: {class_label}, Domain: {domain_label}")
plt.axis('off')
plt.show()
