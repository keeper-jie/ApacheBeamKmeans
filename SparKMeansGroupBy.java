package kmeans;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

//k-means by spark group by
public final class SparKMeansGroupBy {
    static Broadcast<List<String>> init_center_broadcast;

    public static void main(String[] args) throws Exception {

        //execution time
        double startTime = System.nanoTime();
        String inputFile = "";
        String output = "";
        String time_file = "";
        String iteration_file = "";

        //parameter
        for (int i = 0; i < args.length; i++) {
            String[] config_remove_dash = args[i].split("--");
            String[] config = config_remove_dash[1].split("=");

            if (config.length != 2) {
                System.out.println("input format error. Usage:--inputFile=[input_file_path] --output=[output_file_path] --k=[center_count] --time_file=[time_file_path] --iteration_file=[iteration_file_path] --paradigm=[paradigm(optional)] --threshold=[threshold(optional)] --max_iteration=[max_iteration(optional)]");
            }
            if (config[0].equals("inputFile")) {
                inputFile = config[1];
            } else if (config[0].equals("output")) {
                output = config[1];
            } else if (config[0].equals("k")) {
                Point.k = Integer.parseInt(config[1]);
            } else if (config[0].equals("paradigm")) {
                Point.paradigm = Integer.parseInt(config[1]);
            } else if (config[0].equals("threshold")) {
                Point.threshold = Double.parseDouble(config[1]);
            } else if (config[0].equals("max_iteration")) {
                Point.max_iteration = Integer.parseInt(config[1]);
            } else if (config[0].equals("time_file")) {
                time_file = config[1];
            } else if (config[0].equals("iteration_file")) {
                iteration_file = config[1];
            }
        }

        int experiment_time = 10;
        for (int experiment_i = 0; experiment_i < experiment_time; experiment_i++) {
            SparkSession spark = SparkSession
                    .builder()
                    .appName("SparKMeansGroupBy")
                    .getOrCreate();
            JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());

            JavaPairRDD<String, Iterable<String>> center_data_group_by = null;
            //read file and take data
            final JavaRDD<String> data = spark.read().textFile(inputFile).javaRDD().cache();
            //take sample center
            List<String> init_center = data.takeSample(false, Point.k);
            init_center_broadcast = jsc.broadcast(init_center);

//        boolean stop = false;
            int n = 0;
            while (true) {
//            for (int j = 0; j < Point.max_iteration; j++) {
//            List<String> test_data = data.collect();
                //data assigned to center
                JavaPairRDD<String, String> data_assign_center = data.mapToPair(((PairFunction<String, String, String>) p -> {
                    if (p.length() != 0) {
                        //spark null error in standalone
                        Point map_data = Point.StringToPoint(p);
                        List<String> init_center_list = init_center_broadcast.value();
                        double min_distance = Double.MAX_VALUE;
                        Point nearest_point = new Point();

                        //calculate the distance with each data and ini_center
                        for (int i = 0; i < init_center_list.size(); i++) {
                            Point center = Point.StringToPoint(init_center_list.get(i));
                            double distance = Point.distance_paradigm(map_data, center, Point.paradigm);
                            if (distance < min_distance) {
                                min_distance = distance;
                                nearest_point = center;
                            }
                        }
                        return new Tuple2(nearest_point.toString(), p);
                    } else {
                        return null;
                    }
                }));
//            List<Tuple2<String, String>> test_data_assign_center = data_assign_center.collect();

                //group_by_key
                center_data_group_by = data_assign_center.groupByKey();
                //calculate the new center
                JavaRDD<String> new_center = center_data_group_by.map((Function<Tuple2<String, Iterable<String>>, String>) t -> {
                    int count = 0;
                    Point p = new Point();
                    for (String s : t._2) {
                        Point s_point = Point.StringToPoint(s);
                        p = Point.add(p, s_point);
                        count++;
                    }
                    Point result = Point.divide(p, count);
                    return result.toString();
                });

//                init_center = new_center.takeSample(false, Point.k);

                boolean stop = true;
                List<String> new_center_collect = new_center.collect();
                //whether new center equal old center or not
                for (int i = 0; i < new_center_collect.size(); i++) {
                    //calculate init_center whether equal new center or not
                    List<String> init_center_list = init_center_broadcast.getValue();
                    //default value is true,if any center do not equal return false
                    Point new_center_point = Point.StringToPoint(new_center_collect.get(i));
                    boolean is_equal = false;
                    //compare the new center with old center,default is false,if any ont equal,return true
                    for (int j = 0; j < init_center_list.size(); j++) {
                        Point old_center_point = Point.StringToPoint(init_center_list.get(i));
                        double distance = Point.distance_paradigm(new_center_point, old_center_point, Point.paradigm);
                        if (distance < Point.threshold) {
                            is_equal = true;
                            break;
                        }
                    }
                    //if is_equal=false,stop=false
                    if (!is_equal) {
                        stop = false;
                        break;
                    }
                }
                n++;
                if (!stop && n < Point.max_iteration) {
                    //update the data
                    init_center = new_center.takeSample(false, Point.k);
                    init_center_broadcast.unpersist(true);
                    init_center_broadcast = jsc.broadcast(init_center);
                } else {
                    break;
                }
            }
            //write center and data assigned
            for (Tuple2<String, Iterable<String>> t2 : center_data_group_by.collect()) {
                String result = "";
                result += t2._1 + ":";
                for (String s : t2._2()) {
                    result += s + ";";
                }
                write_file(output + experiment_i, result, true);
            }

            //write center
//        for (String p : init_center_broadcast.value()) {
//            System.out.println(p.toString());
////            write_file(output, p.toString(), true);
//        }

            spark.stop();
            String path = time_file;
            double endTime = System.nanoTime();
            double duration = (endTime - startTime) / 1000000000;
            write_file(path, duration + "", true);
            System.out.println("TIME:" + duration);
            System.out.println("Iteration:" + n);
            write_file(iteration_file, n + "", true);
        }
    }

    //write file
    public static void write_file(String path, String value, boolean is_append) throws IOException {
        FileWriter writer = new FileWriter(path, is_append);
        writer.write(value + System.lineSeparator());
        writer.close();
    }
}

