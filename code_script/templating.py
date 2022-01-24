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
                    # [tokenizer.mask_token, tokenizer.mask_token, tokenizer.mask_token],
                    ['was', tokenizer.mask_token, 'to'],
                    ['the', tokenizer.mask_token],
             ]
            print (i)
            # info['labels'] = [
            #     (i[2],),
            #     (i[3],i[4],i[5]),
            #     (i[6],)
            # ]
            tmp = 'irrelevant' if info['name'] == 'Other' else 'relevant'
            if i[0] == '1':
                info['labels'] = [
                    (i[2],),
                    (tmp,),
                    (i[3],)
                ]
            elif i[0] == '2':
                info['labels'] = [
                    (i[3],),
                    (tmp,),
                    (i[2],)
                ]
            else:
                raise ValueError('Invalid identifier in temp.txt', i[0])
            print (info)
            temps[info['name']] = info
    return temps
