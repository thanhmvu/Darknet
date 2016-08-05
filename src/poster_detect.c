#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include "string.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif

char *poster_names[] = {"0","1","2","3","4"};

void train_poster(char *cfgfile, char *weightfile)
{
    char *train_images = "../../database/trainPosters/90C_1kP_s0_train/train.txt";
    char *backup_directory = "../../database/trainPosters/90C_1kP_s0_train/weights_200k";
    srand(time(0));
    data_seed = time(0);
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = net.batch*net.subdivisions;
    int i = *net.seen/imgs;
    data train, buffer;


    layer l = net.layers[net.n - 1];

    int side = l.side;
    int classes = l.classes;
    float jitter = l.jitter;

    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = side;
    args.d = &buffer;
    args.type = REGION_DATA;

    pthread_t load_thread = load_data_in_thread(args);
    clock_t time;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net.max_batches){
        i += 1;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_in_thread(args);

        printf("Loaded: %lf seconds\n", sec(clock()-time));

        time=clock();
        float loss = train_network(net, train);
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, get_current_rate(net), sec(clock()-time), i*imgs);
        if(i%1000==0 || (i < 1000 && i%100 == 0)){
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        free_data(train);
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}

void convert_poster_detections(float *predictions, int classes, int num, int square, int side, int w, int h, float thresh, float **probs, box *boxes, int only_objectness)
{
    int i,j,n;
    //int per_cell = 5*num+classes;
    for (i = 0; i < side*side; ++i){
        int row = i / side;
        int col = i % side;
        for(n = 0; n < num; ++n){
            int index = i*num + n;
            int p_index = side*side*classes + i*num + n;
            float scale = predictions[p_index];
            int box_index = side*side*(classes + num) + (i*num + n)*4;
            boxes[index].x = (predictions[box_index + 0] + col) / side * w;
            boxes[index].y = (predictions[box_index + 1] + row) / side * h;
            boxes[index].w = pow(predictions[box_index + 2], (square?2:1)) * w;
            boxes[index].h = pow(predictions[box_index + 3], (square?2:1)) * h;
            for(j = 0; j < classes; ++j){
                int class_index = i*classes;
                float prob = scale*predictions[class_index+j];
                probs[index][j] = (prob > thresh) ? prob : 0;
            }
            if(only_objectness){
                probs[index][0] = scale;
            }
        }
    }
}

// void print_poster_detections(FILE **fps, char *id, box *boxes, float **probs, int total, int classes, int w, int h)
// {
//     int i, j;
//     for(i = 0; i < total; ++i){
//         float xmin = boxes[i].x - boxes[i].w/2.;
//         float xmax = boxes[i].x + boxes[i].w/2.;
//         float ymin = boxes[i].y - boxes[i].h/2.;
//         float ymax = boxes[i].y + boxes[i].h/2.;

//         if (xmin < 0) xmin = 0;
//         if (ymin < 0) ymin = 0;
//         if (xmax > w) xmax = w;
//         if (ymax > h) ymax = h;

//         for(j = 0; j < classes; ++j){
//             if (probs[i][j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, probs[i][j],
//                     xmin, ymin, xmax, ymax);
//         }
//     }
// }

void validate_poster(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    char *base = "results/comp4_det_test_";
    //list *plist = get_paths("data/voc.2007.test");
		list *plist = get_paths("/home/pjreddie/data/voc/2007_test.txt");
    //list *plist = get_paths("data/voc.2012.test");
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;
    int square = l.sqrt;
    int side = l.side;

    int j;
    FILE **fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j){
        char buff[1024];
        snprintf(buff, 1024, "%s%s.txt", base, poster_names[j]);
        fps[j] = fopen(buff, "w");
    }
    box *boxes = calloc(side*side*l.n, sizeof(box));
    float **probs = calloc(side*side*l.n, sizeof(float *));
    for(j = 0; j < side*side*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;
    int t;

    float thresh = .001;
    int nms = 1;
    float iou_thresh = .5;

    int nthreads = 2;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.type = IMAGE_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    time_t start = time(0);
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            float *predictions = network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
            convert_poster_detections(predictions, classes, l.n, square, side, w, h, thresh, probs, boxes, 0);
            if (nms) do_nms_sort(boxes, probs, side*side*l.n, classes, iou_thresh);
            print_poster_detections(fps, id, boxes, probs, side*side*l.n, classes, w, h);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", (double)(time(0) - start));
}


void test_poster(char *cfgfile, char *weightfile, char *filename, float thresh)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    detection_layer l = net.layers[net.n-1];
    set_batch_network(&net, 1);
    srand(2222222);
    clock_t time;
    char buff[256];
    char *input = buff;
    int j;
    float nms=.5;
    box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
    float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
        image sized = resize_image(im, net.w, net.h);
        float *X = sized.data;
        time=clock();
        float *predictions = network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        convert_poster_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, thresh, probs, boxes, 0);
      
        if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, l.classes, nms);
//         draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, poster_names, voc_labels, l.classes);
        save_image(im, "predictions");
        show_image(im, "predictions");

        show_image(sized, "resized");
        free_image(im);
        free_image(sized);
#ifdef OPENCV
        cvWaitKey(0);
        cvDestroyAllWindows();
#endif
        if (filename) break;
    }
}

void print(int num, float thresh, float **probs, char **names, int classes)
{
    int i;
    for(i = 0; i < num; ++i){
        int class = max_index(probs[i], classes);
        float prob = probs[i][class];
        if(prob > thresh){
            printf("%s: %.0f%%\n", names[class], prob*100);
        }
    }
}

int getResults(int num, float thresh, float **probs, int classes){
	float max_prob = 0;
	int max_class = -1;
	float max_prob2 = 0;
	int max_class2 = -1;
	int j;
	for(j = 0; j < num; ++j){
		int class = max_index(probs[j], classes);
		float prob = probs[j][class];
		if(prob > thresh){
			if (prob> max_prob){
				max_prob = prob;
				max_class = class;
			}else if (prob> max_prob2){
				max_prob2 = prob;
				max_class2 = class;
			}
		}
	}
	printf("1st poster: %d - %.0f%%\n",  max_class, max_prob*100);
	printf("2nd poster: %d - %.0f%%\n",  max_class2, max_prob2*100);
	printf("\n");
	return max_class;
}

void test_posReg(char *cfgfile, char *weightfile, char *folder, float thresh, int N, int ipp)
{
	// Create the deep net
	network net = parse_network_cfg(cfgfile);
	if(weightfile){
		load_weights(&net, weightfile);
	}
	detection_layer l = net.layers[net.n-1];
	set_batch_network(&net, 1);
	srand(2222222);
	clock_t time;
	char buff[256];
	char *input = buff;
	int j;
	float nms=.5;
	box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
	float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
	for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));

	// Loop through and query all test images
	int i;
	int correct = 0;
	for(i = 0; i < N; i++){
		char idx[16];
		snprintf(idx, sizeof(idx), "%06d", i);
		char filename[64];
		strcpy(filename,folder);
		strcat(filename,idx);
		strcat(filename,".jpg");
		printf("%s",filename);

		strncpy(input, filename, 256);
		image im = load_image_color(input,0,0);
		image sized = resize_image(im, net.w, net.h);
		float *X = sized.data;
		time=clock();
		float *predictions = network_predict(net, X);
		printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
		convert_poster_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, thresh, probs, boxes, 0);
		if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, l.classes, nms);
		
		// print results and update number of correct posters
		int max_class = getResults(l.side *l.side *l.n, thresh, probs, l.classes);
		int true_class = i /ipp;
		if (true_class == max_class){ correct++; }
	}
	printf("Accuracy: %.0f%%\n", correct* 100.0/N);
}

int get_poster_class(char * path){
	int ipp = 100; // number of test images per poster
  // methods to extract the class from the path
	char* copy = malloc (1 + strlen(path));
	strcpy(copy,path);
	char* dim = "/."; // divide string by "/" and "."
	char* iterator = strtok(copy,dim);
  char* index = "-1";
	while(strcmp(iterator,"jpg") != 0){
	    index = iterator;
	    iterator = strtok(NULL,dim);
	}
	int class = atoi(index)/ipp;
  return class;
}

int updateCorrect(int num, float thresh, float **probs, int classes,char * path, int correct){
    float max_prob = 0; float max_prob2 = 0;
    int max_class = -1; int max_class2 = -1;
    int j;
    for(j = 0; j < num; ++j){
        int class = max_index(probs[j], classes);
        float prob = probs[j][class];
        if(prob > thresh){
            if (prob> max_prob){
                max_prob = prob;
                max_class = class;
            } else if (prob> max_prob2) {
                max_prob2 = prob;
                max_class2 = class; } } }
    
    // search for max_class in the ground truth
    if (max_class != get_poster_class(path)){ 
      printf("==] THANH: image %s\n", path);
      printf("1st poster: %d - %.0f%%\n",  max_class, max_prob*100);
      printf("2nd poster: %d - %.0f%%\n",  max_class2, max_prob2*100);
    }else{ correct++; }
    return correct;
}

void validate_posReg(char *cfgfile, char *weightfile, char * filename)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

//     char *base = "results/comp4_det_test_";
    //list *plist = get_paths("data/voc.2007.test");
		char * path = (filename != 0) ? filename: "../../database/testPosters/0_90_100/test.txt";
		list *plist = get_paths(path);
    //list *plist = get_paths("data/voc.2012.test");
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;
    int square = l.sqrt;
    int side = l.side;

    int j;
//     FILE **fps = calloc(classes, sizeof(FILE *));
//     for(j = 0; j < classes; ++j){
//         char buff[1024];
//         snprintf(buff, 1024, "%s%s.txt", base, poster_names[j]);
//         fps[j] = fopen(buff, "w");
//     }
    box *boxes = calloc(side*side*l.n, sizeof(box));
    float **probs = calloc(side*side*l.n, sizeof(float *));
    for(j = 0; j < side*side*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;
    int t;

    float thresh = .001;
    int nms = 1;
    float iou_thresh = .5;

    int nthreads = 2;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.type = IMAGE_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    time_t start = time(0);
	
    int total = 0;
    int correct = 0;
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            float *predictions = network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
            convert_poster_detections(predictions, classes, l.n, square, side, w, h, thresh, probs, boxes, 0);
            if (nms) do_nms_sort(boxes, probs, side*side*l.n, classes, iou_thresh);
//             print_poster_detections(fps, id, boxes, probs, side*side*l.n, classes, w, h);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);

            // get accuracy of classification
            // [!] the method to extract the truth class is hardcored in updateCorrect
            correct = updateCorrect(side *side *l.n, thresh, probs, classes,path, correct);
            total++;
          
            printf("Current classification accuracy: %.02f%%\n", correct*100.0/total);	
        }
    }
    printf("Final classification accuracy: %.02f%%\n", correct*100.0/total);
    fprintf(stderr, "Total Detection Time: %f Seconds\n", (double)(time(0) - start));
}

void run_poster(int argc, char **argv)
{
    float thresh = find_float_arg(argc, argv, "-thresh", .2);
    int N = find_float_arg(argc, argv, "-n", 1);
    int ipp = find_float_arg(argc, argv, "-ipp", 1000);
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *filename = (argc > 5) ? argv[5]: 0;
    if(0==strcmp(argv[2], "test")) test_poster(cfg, weights, filename, thresh);
    else if(0==strcmp(argv[2], "train")) train_poster(cfg, weights);
    else if(0==strcmp(argv[2], "valid")) validate_poster(cfg, weights);
    else if(0==strcmp(argv[2], "testpr")) test_posReg(cfg, weights, filename, thresh, N, ipp);
    else if(0==strcmp(argv[2], "validpr")) validate_posReg(cfg, weights, filename);
}
