from arguments import get_args_parser


def get_temps(tokenizer):
    args = get_args_parser()
    temps = {}
    with open(args.data_dir + "/" + args.temps, "r") as f:
        for i in f.readlines():
            i = i.strip().split("\t")
            info = {}
            info['name'] = i[1].strip()
            info['temp'] = [
                    ['the', tokenizer.mask_token],
                    [tokenizer.mask_token, tokenizer.mask_token, tokenizer.mask_token],
                    # ['was', tokenizer.mask_token, 'to'],
                    ['the', tokenizer.mask_token],
             ]
            print (i)
            info['labels'] = [
                (i[2],),
                (i[3],i[4],i[5]),
                (i[6],)
            ]
            # tmp = 'irrelevant' if info['name'] == 'Other' else 'relevant'
            # info['labels'] = [
            #     (i[2],),
            #     (tmp,),
            #     (i[3],)
            # ]
            print (info)
            temps[info['name']] = info
    return temps
