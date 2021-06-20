import pickle
import torch
from FasterRCNN_VG import demo
from m2transformer.data import TextField
from m2transformer.models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory

def main(img_path):
    # ===============Faster RCNN with Visual Genome=============================
    # img_path = 'test_imgs/test1.jpg'

    parser = demo.parse_args()
    parser.add_argument('--ft_path', default='models/ft.pkl')
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='FasterRCNN_VG/cfgs/res101.yml', type=str)
    parser.add_argument('--classes_dir', dest='classes_dir', help='directory to load object classes for classification', default="FasterRCNN_VG/data/genome/1600-400-20")
    parser.add_argument('--net', help='vgg16, res50, res101, res152', default='res101')
    parser.add_argument('--dataset', default='vg')
    parser.add_argument('--image_dir', default=img_path)
    parser.add_argument('--load_dir', help='directory to load models', default='models/faster_rcnn_res101_vg.pth')
    parser.add_argument('-f')
    args = parser.parse_args()

    ft_output = demo.main(args)

    # ===============M2 Transformer=============================
    def predict_caption(model, image_ft, text_field):
        model.eval()
        image_ft = image_ft.to(device)
        with torch.no_grad():
            out, _ = model.beam_search(image_ft, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
        caps_gen = text_field.decode(out, join_words=False)
        return caps_gen


    device = torch.device('cpu')

    # Pipeline  for image regions
    # image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=50, load_in_tmp=False)

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                            remove_punctuation=True, nopoints=False)
    # text_field.vocab = pickle.load(open('vocab_m2_transformer.pkl', 'rb'))
    import sys
    sys.path.insert(0, 'm2transformer')
    text_field.vocab = pickle.load(open('m2transformer/vocab.pkl', 'rb'))

    # Model and dataloaders
    encoder = MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttentionMemory,
                                        attention_module_kwargs={'m': 40})
    decoder = MeshedDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

    if torch.cuda.is_available():
        data = torch.load('models/meshed_memory_transformer.pth')
    else:
        data = torch.load('models/meshed_memory_transformer.pth', map_location=torch.device('cpu'))
    # data = torch.load('saved_models/m2_transformer_temp.pth')
    model.load_state_dict(data['state_dict'])

    # feature extraction
    with open('models/ft.pkl', 'rb') as f:
        ft_output = pickle.load(f)
        ft_output = ft_output.unsqueeze(0)
    print(ft_output.shape)
    # img captioning
    cap = predict_caption(model, ft_output, text_field)
    return cap