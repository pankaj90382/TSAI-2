import annotated_encoder_decoder_de_en
import en_core_web_sm
import de_core_news_sm
import torch
import model as annotated_encoder_decoder_de_en

def translate_annotated_encoder_decoder_de_en(model,meta,source_text):

    spacy_de = de_core_news_sm.load()

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    src_tok = tokenize_de(source_text)

    src_idx = [meta["SRC.vocab.stoi"][x] for x in src_tok] + [
        meta["SRC.vocab.stoi"][meta["EOS_TOKEN"]]
    ]
    src = torch.LongTensor(src_idx)
    src_mask = (src != meta["SRC.vocab.stoi"][meta["PAD_TOKEN"]]).unsqueeze(-2)
    src_length = torch.tensor(len(src))

    # convert to batch size 1
    src = src.unsqueeze(0)
    src_mask = src_mask.unsqueeze(0)
    src_length = src_length.unsqueeze(0)

    output = annotated_encoder_decoder_de_en.greedy_decode(
        model,
        src,
        src_mask,
        src_length,
        max_len=100,
        sos_index=meta["TRG.vocab.stoi"][meta["SOS_TOKEN"]],
        eos_index=meta["TRG.vocab.stoi"][meta["EOS_TOKEN"]],
    )

    return " ".join([meta["TRG.vocab.itos"][x] for x in output])

def get_text_translate_function(source_text):
        meta_file = ''
        model_file = ''
        meta = load_meta_dill(meta_file)
        model_state = torch.load(
            model_file, map_location="cpu"
        )

        model: annotated_encoder_decoder_de_en.EncoderDecoder = (
            annotated_encoder_decoder_de_en.make_model(
                len(meta["SRC.vocab.itos"]),
                len(meta["TRG.vocab.itos"]),
                emb_size=256,
                hidden_size=256,
                num_layers=1,
                dropout=0.2,
            )
        )
        model.load_state_dict(model_state)
		return translate_annotated_encoder_decoder_de_en(model, meta, source_text)
		
def load_meta_dill(path):
    import dill

    inp = open(path, "rb")
    meta = dill.load(inp)
    inp.close()

    return meta