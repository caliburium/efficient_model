import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from dataloader.DomainNetLoader import dn_loader
from model.AlexNet import AlexNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.90)
    parser.add_argument('--opt_decay', type=float, default=1e-6)
    args = parser.parse_args()

    num_epochs = args.epoch
    # Initialize Weights and Biases
    wandb.init(entity="hails",
               project="Efficient Model",
               config=args.__dict__,
               name="[Alex]DNet_" + "_lr:" + str(args.lr) + "_Batch:" + str(args.batch_size)
               )

    # quickdraw, real, painting, sketch, clipart, infograph
    # furniture ~ 246 table, 110 teapot, 15 streetlight, 213 umbrella, 139 wine glass, 299 stairs, 58 toothbrush, 102 suitcase, 47 ladder, 48 picture frame
    furniture_real_train, furniture_real_test = dn_loader('real', [246, 110, 15, 213, 139, 299, 58, 102, 47, 48], args.batch_size)
    furniture_quickdraw_train, furniture_quickdraw_test = dn_loader('quickdraw', [246, 110, 15, 213, 139, 299, 58, 102, 47, 48], args.batch_size)

    # mammal ~ 61 squirrel, 292 dog, 81 whale, 148 tigger, 319 zebra, 157 sheep, 83 elephant, 188 horse, 312 cat, 89 monkey
    # mammal_real_train, mammal_real_test = dn_loader('real', [61, 292, 81, 148, 319, 157, 83, 188, 312, 89], args.batch_size)
    # mammal_paint_train, mammal_paint_test = dn_loader('painting', [61, 292, 81, 148, 319, 157, 83, 188, 312, 89], args.batch_size)

    # tool ~ 314 nail, 131 sword, 227 bottlecap, 12 basket, 40 rifle, 249 bandage, 10 pliers, 237 axe, 207 paint can, 276 anvil
    tool_real_train, tool_real_test = dn_loader('real', [314, 131, 227, 12, 40, 249, 10, 237, 207, 276], args.batch_size)
    # tool_painting_train, tool_painting_test = dn_loader('painting', [314, 131, 227, 12, 40, 249, 10, 237, 207, 276], args.batch_size)

    print("Data load complete, start training")

    model = AlexNet(pretrained=False, progress=True, num_class=10).to(device)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.opt_decay)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        i = 0
        total_furniture_quickdraw_loss, total_furniture_real_loss, total_tool_real_loss, total_loss = 0, 0, 0, 0
        total_furniture_quickdraw_correct, total_furniture_real_correct, total_tool_real_correct = 0, 0, 0
        furniture_quickdraw_samples, furniture_real_samples, tool_real_samples = 0, 0, 0

        for furniture_quickdraw_data, furniture_real_data, tool_real_data in zip(furniture_quickdraw_train, furniture_real_train, tool_real_train):
            furniture_quickdraw_images, furniture_quickdraw_labels = furniture_quickdraw_data
            furniture_quickdraw_images, furniture_quickdraw_labels = furniture_quickdraw_images.to(device), furniture_quickdraw_labels.to(device)
            furniture_real_images, furniture_real_labels = furniture_real_data
            furniture_real_images, furniture_real_labels = furniture_real_images.to(device), furniture_real_labels.to(device)
            tool_real_images, tool_real_labels = tool_real_data
            tool_real_images, tool_real_labels = tool_real_images.to(device), tool_real_labels.to(device)

            furniture_quickdraw_out = model(furniture_quickdraw_images)
            furniture_real_out = model(furniture_real_images)
            tool_real_out = model(tool_real_images)

            furniture_quickdraw_loss = criterion(furniture_quickdraw_out, furniture_quickdraw_labels)
            furniture_real_loss = criterion(furniture_real_out, furniture_real_labels)
            tool_real_loss = criterion(tool_real_out, tool_real_labels)
            loss = furniture_quickdraw_loss + furniture_real_loss + tool_real_loss

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_furniture_quickdraw_loss += furniture_quickdraw_loss.item() * furniture_quickdraw_labels.size(0)
            total_furniture_real_loss += furniture_real_loss.item() * furniture_real_labels.size(0)
            total_tool_real_loss += tool_real_loss.item() * tool_real_labels.size(0)
            total_loss += loss.item() * (furniture_quickdraw_labels.size(0) + furniture_real_labels.size(0) + tool_real_labels.size(0))

            furniture_quickdraw_correct = (torch.argmax(furniture_quickdraw_out, dim=1) == furniture_quickdraw_labels).sum().item()
            furniture_real_correct = (torch.argmax(furniture_real_out, dim=1) == furniture_real_labels).sum().item()
            tool_real_correct = (torch.argmax(tool_real_out, dim=1) == tool_real_labels).sum().item()

            total_furniture_quickdraw_correct += furniture_quickdraw_correct
            total_furniture_real_correct += furniture_real_correct
            total_tool_real_correct += tool_real_correct

            furniture_quickdraw_samples += furniture_quickdraw_labels.size(0)
            furniture_real_samples += furniture_real_labels.size(0)
            tool_real_samples += tool_real_labels.size(0)

            i += 1

        furniture_quickdraw_acc_epoch = total_furniture_quickdraw_correct / furniture_quickdraw_samples * 100
        furniture_real_acc_epoch = total_furniture_real_correct / furniture_real_samples * 100
        tool_real_acc_epoch = total_tool_real_correct / tool_real_samples * 100

        furniture_quickdraw_avg_loss = total_furniture_quickdraw_loss / furniture_quickdraw_samples
        furniture_real_avg_loss = total_furniture_real_loss / furniture_real_samples
        tool_real_avg_loss = total_tool_real_loss / tool_real_samples
        total_avg_loss = total_loss / (furniture_quickdraw_samples + furniture_real_samples + tool_real_samples)

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], | '
            f'Total Loss: {total_avg_loss:.4f}, | '
            f'Furniture Quickdraw Loss: {furniture_quickdraw_avg_loss:.4f}, | '
            f'Furniture Real Loss: {furniture_real_avg_loss:.4f}, | '
            f'Tool Real Loss: {tool_real_avg_loss:.4f} | '
        )

        print(
            f'Furniture Quickdraw Acc: {furniture_quickdraw_acc_epoch:.3f}% | '
            f'Furniture Real Acc: {furniture_real_acc_epoch:.3f}% | '
            f'Tool Real Acc: {tool_real_acc_epoch:.3f}% | '
        )

        wandb.log({
            'Train/Label Loss': total_avg_loss,
            'Train/Furniture Quickdraw Label Loss': furniture_quickdraw_avg_loss,
            'Train/Furniture Real Label Loss': furniture_real_avg_loss,
            'Train/Tool Real  Label Loss': tool_real_avg_loss,
            'Train/Furniture Quickdraw Label Accuracy': furniture_quickdraw_acc_epoch,
            'Train/Furniture Real Label Accuracy': furniture_real_acc_epoch,
            'Train/Tool Real  Label Accuracy': tool_real_acc_epoch,
        }, step=epoch)

        model.eval()

        def tester(loader, group):
            correct, total = 0, 0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)

                class_output = model(images)
                total += labels.size(0)
                correct += (torch.argmax(class_output, dim=1) == labels).sum().item()

            accuracy = correct / total * 100
            wandb.log({f'Test/Label {group} Accuracy': accuracy}, step=epoch)
            print(f'Test {group} | Label Acc: {accuracy:.3f}%')

        with torch.no_grad():
            tester(furniture_quickdraw_test, 'Furniture Quickdraw')
            tester(furniture_real_test, 'Fruniture Real')
            tester(tool_real_test, 'Tool Real')

if __name__ == '__main__':
    main()
