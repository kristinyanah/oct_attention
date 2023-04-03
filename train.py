#!/usr/bin/python
import argparse
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import torchvision.transforms as T
from data import get_data
from losses import CombinedLoss_adv2, FocalFrequencyLoss
from networks import get_model
from utils import per_class_dice, per_class_dice_mask


def argument_parser():
    parser = argparse.ArgumentParser()

    # Optimization hyperparameters
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--num_iterations', default=250, type=int)
    #parser.add_argument('--learning_rate', default=5e-4, type=float)
    parser.add_argument('--learning_rate', default=3e-4, type=float)
    parser.add_argument('--n_classes', default=9, type=int)
    parser.add_argument('--ffc_lambda', default=0, type=float)
     
    
    #parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    # Dataset options
    parser.add_argument('--dataset', default='Duke', choices=["Duke", "UMN"])
    parser.add_argument('--image_size', default='224', type=int)

    parser.add_argument('--image_dir', default="./DukeData/")
    parser.add_argument('--model_name', default="y_net_layer_mp2", choices=["unet", "y_net_gen", "y_net_gen_ffc", "y_net_gen_advance", "y_net_gen_advance_gcn", \
        "y_net_gen_advance2", "y_net_gen_advance2_gcn",  "y_net_gen_advance_double", "y_net_gen_advance2_cat", "y_net_gen_advance2_cat_double", "y_net_gen_advance2_branch", \
            "y_net_gen_advance2_branch_double", "y_net_gen_advance2_combine", "y_net_gen_advance2_double_graph", "y_net_gen_advance2_cross", "y_net_gen_advance2_double_graph_true", \
                "y_net_gen_ffc_cs", "y_net_gen_ffc_cs_cat", "y_net_gen_cat_cs", "y_net_two_cs", "y_net_gen_cat_cs_channel", "unetatt", "y_net_gen_cat_cs_final", "y_net_add_att_in", \
                    "y_net_gen_att_cross", "y_net_two_cs", "y_net_layer_add", "y_net_layer_mp2"])

    # Network options
    parser.add_argument('--g_ratio', default=0.5, type=float)

    # Other options
    parser.add_argument('--device', default="cuda", choices=["cuda", "cpu"])
    parser.add_argument('--seed', default=7, type=int)

    return parser


def colored_text(st):
    return '\033[91m' + st + '\033[0m'


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def eval_mask(val_loader, criterion, model, n_classes, dice_s=True, device="cuda", im_save=False):
    model.eval()
    loss = 0
    counter = 0
    dice = 0

    dice_all = np.zeros(n_classes)

    for img, label in tqdm.tqdm(val_loader):
        img = img.to(device)
        img = torch.cat((img, img), dim=0)
        label = label.to(device)
        label_new = label[0]
        label = torch.cat((label, label), dim=0)
        label_oh = torch.nn.functional.one_hot(label, num_classes=n_classes)
         
        
        transform = T.Resize((128,128))
        label_new = transform(label_new)
        label_oh_new = torch.nn.functional.one_hot(label_new, num_classes=n_classes).squeeze()#[224, 224, 10])
        label_oh_new = label_oh_new.permute(2, 0, 1) #[9, 224, 224])
        pred = model(img)
        max_val, idx = torch.max(pred['original'], 1)
        pred_oh = torch.nn.functional.one_hot(idx, num_classes=n_classes)


        b_label_new = label
        b_label_new = transform(b_label_new) #[2, 1, 128, 128])
        b_label_oh_new = torch.nn.functional.one_hot(b_label_new, num_classes=n_classes).squeeze()
        b_label_oh_new = b_label_oh_new.permute(0, 3, 1, 2) 


        if dice_s:
            d1, d2 = per_class_dice_mask(pred_oh, label_oh, n_classes)
            dice += d1
            dice_all += d2
        #loss += criterion(pred['original'], label.squeeze(1), pred['pred_mask'], b_label_oh_new, b_label_new.squeeze(1), device=device).item()
        loss += criterion(pred['original'], pred['pred_mask'], label.squeeze(1), device=device).item()
 

        counter += 1

    loss = loss / counter
    dice = dice / counter
    dice_all = dice_all / counter
    print("Validation loss: ", loss, " Mean Dice: ", dice.item(), "Dice All:", dice_all)
    return dice
#改
#改
#改
##target = {}
#target['labels'] = [torch.zeros((7), dtype=torch.long).cuda(), torch.ones((3), dtype=torch.long).cuda()]
#target['masks'] = [torch.zeros((7, 128, 128)).cuda(), torch.ones((3, 128, 128)).cuda()]

#origin
def eval(val_loader, criterion, model, n_classes, dice_s=True, device="cuda", im_save=False):
    model.eval()
    loss = 0
    counter = 0
    dice = 0

    dice_all = np.zeros(n_classes)

    for img, label in tqdm.tqdm(val_loader):
        img = img.to(device)
        label = label.to(device)
        label_oh = torch.nn.functional.one_hot(label, num_classes=n_classes)

        pred = model(img)
        max_val, idx = torch.max(pred, 1)
        pred_oh = torch.nn.functional.one_hot(idx, num_classes=n_classes)

        if dice_s:
            d1, d2 = per_class_dice(pred_oh, label_oh, n_classes)
            dice += d1
            dice_all += d2

        loss += criterion(pred, label.squeeze(1), device=device).item()

        counter += 1

    loss = loss / counter
    dice = dice / counter
    dice_all = dice_all / counter
    print("Validation loss: ", loss, " Mean Dice: ", dice.item(), "Dice All:", dice_all)
    return dice


#trainmask
def train_mask(args):
    device = args.device
    n_classes = args.n_classes
    model_name = args.model_name
    learning_rate = args.learning_rate
    ratio = args.g_ratio
    data_path = args.image_dir
    iterations = args.num_iterations
    img_size = args.image_size
    batch_size = args.batch_size

    criterion_seg = CombinedLoss_adv2()
    criterion_ffc = FocalFrequencyLoss()

    save_name = model_name + ".pt"

    max_dice = 0
    best_test_dice = 0
    best_iter = 0

    model = get_model(model_name, ratio=ratio, num_classes=n_classes).to(device)
    
    
    #reload############################
   
    model1 = get_model('y_net_layer_add', ratio=ratio, num_classes=n_classes).to(device)
    ckpt = torch.load('best_epoch 68__y_net_layer_add.pt')
    model1.load_state_dict(ckpt)
    model_dict = model.state_dict()
    model1_dict = {k: v for k, v in model1.state_dict().items() if k in model_dict}
    model_dict.update(model1_dict)
 
 ################################################################   
    model.train()
    optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate,
                                 weight_decay=args.weight_decay)

    train_loader, val_loader, test_loader, _, _, _ = get_data(data_path, img_size, batch_size)
    #def _get_binary_mask(target):
        # 返回每类的binary mask
        # y = target.size(1)
       #  x = target.size(2)
       #  target_onehot = torch.zeros(9 + 1, y, x) #[10, 224, 224]) [2, 224, 224, 10])
       #  target_onehot = target_onehot.scatter(dim=0, index=target.unsqueeze(0), value=1)
        # return target_onehot[1:]
    
    
    for t in range(iterations):
        for img, label in tqdm.tqdm(train_loader):
            img = img.to(device)
            label = label.to(device) #e([2, 1, 224, 224])
            
            label_new = label[0]
            transform = T.Resize((128,128))
            label_new = transform(label_new) #([1, 128, 128])
            label_oh = torch.nn.functional.one_hot(label, num_classes=n_classes).squeeze() #orch.Size([9, 128, 128])
            label_oh_new = torch.nn.functional.one_hot(label_new, num_classes=n_classes).squeeze()#[torch.Size([9, 128, 128])
            label_oh_new = label_oh_new.permute(2, 0, 1) #torch.Size([9, 128, 128])
            
            
            b_label_new = label
            b_label_new = transform(b_label_new) #[2, 1, 128, 128])
            b_label_oh_new = torch.nn.functional.one_hot(b_label_new, num_classes=n_classes).squeeze()
            b_label_oh_new = b_label_oh_new.permute(0, 3, 1, 2) 
            
            
            #x_resized = F.interpolate(label_oh_new, size=(128, 128), mode='bilinear', align_corners=True)
            #gt_binary_mask = _get_binary_mask(label_new)
            optimizer.zero_grad()
#note
            pred = model(img) #([2, 9, 224, 224])  .2302, 0.1766, 0.0804,  ..., 0.1785, 0.313
            max_val, idx = torch.max(pred['original'], 1)
            pred_oh = torch.nn.functional.one_hot(idx, num_classes=n_classes)
            pred_oh = pred_oh.permute(0, 3, 1, 2)
            label_oh = label_oh.permute(0, 3, 1, 2)
            #loss = criterion_seg(pred['original'], label.squeeze(1), device=device) + args.ffc_lambda * criterion_ffc(pred_oh, label_oh) + pred['maskloss']  #mask loss )
            #找pred #[9, 128, 128]
            #loss = criterion_seg(pred['original'], label.squeeze(1), pred['pred_mask'], b_label_oh_new, b_label_new.squeeze(1), device=device) + args.ffc_lambda * criterion_ffc(pred_oh, label_oh)  
          
            loss = criterion_seg(pred['original'], pred['pred_mask'], label.squeeze(1), device=device)
          
          
          
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if t % 2 == 0:
            print(loss.item())
            
   
        if t % 1 == 0 or t > 45:
            print("Epoch", t, "/", iterations) 
           # print("Validation")
            #dice = eval_mask(val_loader, criterion_seg, model, n_classes=n_classes)
            print("Expert 1 - Test")
            dice_test = eval_mask(test_loader, criterion_seg, model, dice_s=True, n_classes=n_classes)

            if dice > max_dice:
                max_dice = dice
                best_iter = t
                best_test_dice = dice_test
                print(colored_text("Updating model, epoch: "), t)

            torch.save(model.state_dict(), "none/epoch %d__%s"%(int(t), save_name))
            model.train()
    print("Best iteration: ", best_iter, "Best val dice: ", max_dice, "Best test dice: ", best_test_dice)
    return model




def train(args):
    device = args.device
    n_classes = args.n_classes
    model_name = args.model_name
    learning_rate = args.learning_rate
    ratio = args.g_ratio
    data_path = args.image_dir
    iterations = args.num_iterations
    img_size = args.image_size
    batch_size = args.batch_size

    criterion_seg = CombinedLoss()
    criterion_ffc = FocalFrequencyLoss()

    save_name = model_name + ".pt"

    max_dice = 0
    best_test_dice = 0
    best_iter = 0

    model = get_model(model_name, ratio=ratio, num_classes=n_classes).to(device)
    model.train()

    optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate,
                                 weight_decay=args.weight_decay)

    train_loader, val_loader, test_loader, _, _, _ = get_data(data_path, img_size, batch_size)

    for t in range(iterations):
        for img, label in tqdm.tqdm(train_loader):
            img = img.to(device)
            label = label.to(device)
            label_oh = torch.nn.functional.one_hot(label, num_classes=n_classes).squeeze()

            pred = model(img)
            max_val, idx = torch.max(pred, 1)
            pred_oh = torch.nn.functional.one_hot(idx, num_classes=n_classes)
            pred_oh = pred_oh.permute(0, 3, 1, 2)
            label_oh = label_oh.permute(0, 3, 1, 2)
            loss = criterion_seg(pred, label.squeeze(1), device=device) + args.ffc_lambda * criterion_ffc(pred_oh,
                                                                                                          label_oh)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if t % 2 == 0:
            print(loss.item())

        if t % 10 == 0 or t > 45:
            print("Epoch", t, "/", iterations)
            print("Validation")
            dice = eval(val_loader, criterion_seg, model, dice_s=True, n_classes=n_classes)
            print("Expert 1 - Test")
            dice_test = eval(test_loader, criterion_seg, model, n_classes=n_classes)

            if dice > max_dice:
                max_dice = dice
                best_iter = t
                best_test_dice = dice_test
                print(colored_text("Updating model, epoch: "), t)

                torch.save(model.state_dict(), save_name)
            model.train()
    print("Best iteration: ", best_iter, "Best val dice: ", max_dice, "Best test dice: ", best_test_dice)
    return model

if __name__ == "__main__":
    
    args = argument_parser().parse_args()
    print(args)
    set_seed(args.seed)

    train_mask(args)


