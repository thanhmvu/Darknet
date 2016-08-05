#include "network.h"
#include "utils.h"
#include "parser.h"

// char *classify_labels[] = {"zero","one","two","three","four","five","six","seven","eight","nine"};
int classes = 90;

void train_classify(char *cfgfile, char *weightfile)
{
    data_seed = time(0);
    srand(time(0));
    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    char *backup_directory = "../../database/trainPosters/train.90C.1kP_v2/weights_classify_200k/";
    printf("%s\n", base);
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = net.batch;
    int i = *net.seen/imgs;
    list *plist = get_paths("../../database/trainPosters/train.90C.1kP_v2/train.txt");
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    clock_t time;
    while(get_current_batch(net) < net.max_batches){
        ++i;
        time=clock();
        data train = load_data2(paths, imgs, plist->size, classes, net.w, net.h);
        printf("Loaded: %lf seconds\n", sec(clock()-time));

        time=clock();
        float loss = train_network(net, train);
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%d: %f, %f avg, %lf seconds, %d images\n", i, loss, avg_loss, sec(clock()-time), *net.seen);
        free_data(train);
        //if((i % 100) == 0) net.learning_rate *= .1;
        if(i%100==0){
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights",backup_directory,base, i);
            save_weights(net, buff);
        }
    }
}

void validate_classify(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));

    list *plist = get_paths("../../database/testPosters/0_90_100/test.txt");

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    data val = load_data2(paths, m, 0, classes, net.w, net.h);
    float *acc = network_accuracies(net, val, 2);
    printf("Validation Accuracy: %f, %d images\n", acc[0], m);
    free_data(val);
}

void test_classify(char *cfgfile, char *weightfile, char *filename)
{
//     network net = parse_network_cfg(cfgfile);
//     if(weightfile){
//         load_weights(&net, weightfile);
//     }
//     set_batch_network(&net, 1);
//     srand(2222222);
//     int i = 0;
//     char **names = classify_labels;
//     char buff[256];
//     char *input = buff;
//     int indexes[6];
//     while(1){
//         if(filename){
//             strncpy(input, filename, 256);
//         }else{
//             printf("Enter Image Path: ");
//             fflush(stdout);
//             input = fgets(input, 256, stdin);
//             if(!input) return;
//             strtok(input, "\n");
//         }
//         image im = load_image_color(input, net.w, net.h);
//         float *X = im.data;
//         float *predictions = network_predict(net, X);
//         top_predictions(net, 6, indexes);
//         for(i = 0; i < 6; ++i){
//             int index = indexes[i];
//             printf("%s: %f\n", names[index], predictions[index]);
//         }
//         free_image(im);
//         if (filename) break;
//     }
}

void run_poster_classify(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: ./darknet poster_classify [train/test/valid] [cfg] [weights (optional)]\n");
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *filename = (argc > 5) ? argv[5]: 0;
    if(0==strcmp(argv[2], "test")) test_classify(cfg, weights, filename);
    else if(0==strcmp(argv[2], "train")) train_classify(cfg, weights);
    else if(0==strcmp(argv[2], "valid")) validate_classify(cfg, weights);
}

