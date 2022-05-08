package beam_kmeans;

import org.apache.beam.sdk.Pipeline;
import org.apache.beam.sdk.io.TextIO;
import org.apache.beam.sdk.options.Default;
import org.apache.beam.sdk.options.Description;
import org.apache.beam.sdk.options.PipelineOptions;
import org.apache.beam.sdk.options.PipelineOptionsFactory;
import org.apache.beam.sdk.transforms.*;
import org.apache.beam.sdk.values.KV;
import org.apache.beam.sdk.values.PCollection;
import org.apache.beam.sdk.values.PCollectionView;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class BeamKmeansSideInput {
    /**
     * A SimpleFunction that converts <KV<String, Iterable<String>> into a printable string.
     */
    public static class FormatAsTextFn extends SimpleFunction<KV<String, Iterable<String>>, String> {
        @Override
        public String apply(KV<String, Iterable<String>> input) {
            String result = "";
            result += input.getKey() + ":";
            for (String s : input.getValue()) {
                result += s + ";";
            }
            return result;
        }
    }

    /**
     * KMeansOptions set the pipeline's option
     */
    public interface KMeansOptions extends PipelineOptions {
        @Description("Path of the file to read from")
        @Default.String("/share/k-means-mapreduce-master/datasets/1k/dataset_3_7.txt")
        String getInputFile();

        void setInputFile(String value);

        /**
         * Set this required option to specify where to write the output.
         */
        @Description("Path of the file to write to")
        @Default.String("/share/word-count-beam/src/main/java/beam_kmeans/result/spark_kmeans_data.txt")
        String getOutput();

        void setOutput(String value);
    }

    /**
     * Apache Beam K-means
     *
     * @param options   -KMeansOptions
     * @param time_file -the time file path to write
     * @throws IOException
     */
    static void runWordCount(KMeansOptions options, String time_file) throws IOException {
        //execution time
        double startTime = System.nanoTime();
        int experiment_time = 5;
        for (int experiment_i = 0; experiment_i < experiment_time; experiment_i++) {
            PCollection<KV<String, Iterable<String>>> center_data_group_by = null;
            Pipeline p = Pipeline.create(options);

            //input data =>PCollection<String>
            PCollection<String> input_data = p.apply("read_lines", TextIO.read().from(options.getInputFile()));

            //select init_center with K random point string
            PCollection<Iterable<String>> init_center = input_data.apply("take_sample", Sample.fixedSizeGlobally(Point.k));
//        PCollectionView<Iterable<Point>> init_center_view = init_center.apply(View.asSingleton());

            //calculate the distance with init_center and select the max_distance point
            for (int i = 0; i < Point.max_iteration; i++) {
                //broadcast string
                PCollectionView<Iterable<String>> init_center_view = init_center.apply("broad_cast_center", View.asSingleton());

                //data assign to center
                PCollection<KV<String, String>> data_assign_center_sideinput = input_data
                        .apply(ParDo.of(new DoFn<String, KV<String, String>>() {
                            @ProcessElement
                            public void processElement(ProcessContext context) {
                                Point data = Point.StringToPoint(context.element());
                                Iterable<String> center_list = context.sideInput(init_center_view);
                                double min_distance = Double.MAX_VALUE;
                                Point nearest_point = new Point();

                                for (String center : center_list) {
                                    Point center_point = Point.StringToPoint(center);
                                    double distance = Point.distance_paradigm(data, center_point, Point.paradigm);
                                    if (distance < min_distance) {
                                        min_distance = distance;
                                        nearest_point = center_point;
                                    }
                                }
                                context.output(KV.of(nearest_point.toString(), context.element()));
                            }
                        }).withSideInputs(init_center_view));

                //group by the <center,data> by key
                center_data_group_by = data_assign_center_sideinput.apply(GroupByKey.create());

                //output the new center
                PCollection<String> new_center = center_data_group_by.apply(MapElements.via(new SimpleFunction<KV<String, Iterable<String>>, String>() {
                    @Override
                    public String apply(KV<String, Iterable<String>> center_data_iteration) {
                        int count = 0;
                        Point p = new Point();
                        for (String s : center_data_iteration.getValue()) {
                            Point s_point = Point.StringToPoint(s);
                            p = Point.add(p, s_point);
                            count++;
                        }
                        Point result = Point.divide(p, count);
                        return result.toString();
                    }
                }));
                //update the center
                init_center = new_center.apply(Sample.fixedSizeGlobally(Point.k));
            }
            center_data_group_by.apply("format_output", MapElements.via(new FormatAsTextFn()))
                    .apply("write_result_to_file", TextIO.write().withoutSharding().to(options.getOutput() + experiment_i));
            p.run().waitUntilFinish();
            String path = time_file;
            double endTime = System.nanoTime();
            double duration = (endTime - startTime) / 1000000000;
            write_file(path, duration + "", true);
            System.out.println("TIME:" + duration);
        }
    }

    public static void main(String[] args) throws IOException {

        String time_file = "";
        //param
        ArrayList<String> al = new ArrayList<>();
        for (int i = 0; i < args.length; i++) {
            String[] config_remove_dash = args[i].split("--");
            String[] config = config_remove_dash[1].split("=");
            if (config.length != 2) {
                System.out.println("input format error. Usage:--inputFile=[input_file_path] --output=[output_file_path] --runner=[runner] --k=[center_count] --paradigm=[paradigm(optional)] --threshold=[threshold(optional)] --max_iteration=[max_iteration(optional)]");
            }
            if (config[0].equals("k")) {
                Point.k = Integer.parseInt(config[1]);
            } else if (config[0].equals("paradigm")) {
                Point.paradigm = Integer.parseInt(config[1]);
            } else if (config[0].equals("threshold")) {
                Point.threshold = Double.parseDouble(config[1]);
            } else if (config[0].equals("max_iteration")) {
                Point.max_iteration = Integer.parseInt(config[1]);
            } else if (config[0].equals("time_file")) {
                time_file = config[1];
            } else {
                //put the input to arg
                al.add(args[i]);
            }
        }

        args = al.toArray(new String[0]);
        //read init param
        WordCountOptions options =
                PipelineOptionsFactory.fromArgs(args).withValidation().as(WordCountOptions.class);
        runWordCount(options, time_file);
    }

    /**
     * write the String into file
     * @param path -the path write to
     * @param value -the String which will write to file
     * @param is_append -whether append or not
     * @throws IOException
     */
    public static void write_file(String path, String value, boolean is_append) throws IOException {
        FileWriter writer = new FileWriter(path, is_append);
        writer.write(value + System.lineSeparator());
        writer.close();
    }
}
