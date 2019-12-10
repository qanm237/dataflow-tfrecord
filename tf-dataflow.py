'''
    Generate tfrecords for words dataset from csv.
    columns: [gcloud bucket path, label]
'''
import numpy as np
from cv2 import cv2
from face_util import face_landmarks
from hand_util import hand_centre
import argparse
import logging
import gcsfs
import tensorflow as tf
from tf.python.keras.preprocessing.image import load_img
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.io import tfrecordio
import apache_beam as beam

LABEL_DICT = {
    'A LOT OF': 0,
    'AIR': 1,
    'ALSO': 2,
    'AUGUST': 3,
}





def get_args():
  '''
    Can take arguments from command line
    ARGS: None
    Returns: Dictionary for csv-path
  '''
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--csv-path',
      type=str,
      default='gs://bucket_name/folder/csv_file.csv', 
      help='name of csv file')
  return parser.parse_known_args()


class LoadImageDoFn(beam.DoFn):
  '''
  Load image from gcs
  '''
  def __init__(self):
    '''
    Initialization
    '''
    print('begin')

  def process(self, text_line):
    '''
    Load image from gcs
    ARGS: text-line with image path and label separated by semi colon.
          It is yielded from inbuilt class 'beam.io.ReadFromText'
    Yields: (image array, categorical label) tuple
    '''
    print(text_line)
    text_line_list = text_line.split(',')
    path = text_line_list[0]
    fs = gcsfs.GCSFileSystem(
        project='project_name',
        access='full_control')
    label = text_line_list[1]

    image_arr = []
    for image in fs.ls(path):

      with fs.open(image) as img_file:
        img = np.array(load_img(img_file, target_size=(260, 210)))
        image_arr.append(img)

    label = LABEL_DICT[label]
    label = tf.keras.utils.to_categorical(label, 100)
    yield (image_arr, label)


class PreprocessImagesDoFn(beam.DoFn):
  '''
  Transforms every frame into four-stacked(facial-landmarks, left-hand,
    right-hand, rel-pos of hands) images
  '''
  def process(self, image_label_tuple):
    '''
    Transforms frames to four-stacked images by calling different
    functions from other codes
    ARGS: tuble containing original image_array
    YIELD: Four-stacked image_array and labels as tuple
    '''
    images, label = image_label_tuple
    for i in enumerate(images):
      image = images[i]
      detected_face = face_landmarks.detect(image)
      hright, hleft, hcen = hand_centre.detect(image)
      detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
      hright = cv2.cvtColor(hright, cv2.COLOR_BGR2GRAY)
      hleft = cv2.cvtColor(hleft, cv2.COLOR_BGR2GRAY)
      hcen = cv2.cvtColor(hcen, cv2.COLOR_BGR2GRAY)
      stack = np.dstack((detected_face, hright, hleft, hcen))
      stack = np.asarray(stack)
      stack = stack / 255
      images[i] = stack
    yield (images, label)


class ImageToTfExampleDoFn(beam.DoFn):
  '''
  Convert frames-sequence of every word to TFExample
  '''`
  def __init__(self):
    '''
    Initialization
    '''
    print("Running")

  @staticmethod
  def _int64_feature(value):
    '''
    Get int64 feature
    '''
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

  @staticmethod
  def _bytes_feature(value):
    '''
    Get byte features
    '''
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  def process(self, image_label_tuple):
    '''
    Convert frames-sequence of every word to TFExample
    ARGS: Four-stacked image_array and labels as tuple
    YIELD: tf example
    '''
    images, label = image_label_tuple
    feature = {}
    g_labels = label.astype(np.float32)
    feature['label'] = self._bytes_feature(g_labels.tostring())
    feature['len'] = self._int64_feature(len(images))
    feature['height'] = self._int64_feature(60)
    feature['width'] = self._int64_feature(60)
    feature['depth'] = self._int64_feature(4)
    common_features = tf.train.Features(feature=feature)

    image_feature = []
    for image in images:
      raw_image = image.astype(np.float32)
      raw_image = raw_image.tostring()
      raw_image = self._bytes_feature(raw_image)
      image_feature.append(raw_image)

    images_features = tf.train.FeatureList(feature=image_feature)
    images_features1 = tf.train.FeatureLists(
        feature_list={'frames': images_features})
    example = tf.train.SequenceExample(
        context=common_features,
        feature_lists=images_features1)
    yield example


def run_pipeline():
  '''
  Apache beam pipeline
  ARGS: None
  '''
  args, pipeline_args = get_args()
  pipeline_options = PipelineOptions(pipeline_args)
  pipeline_options.view_as(SetupOptions).save_main_session = True

  read_textline_from_csv = beam.io.ReadFromText(
      args.csv_path, skip_header_lines=1)
  load_img_from_path = LoadImageDoFn()

  augment_data = PreprocessImagesDoFn()

  img_to_tfexample = ImageToTfExampleDoFn()

  write_to_tf_record = tfrecordio.WriteToTFRecord(
      file_path_prefix='gs://bucket_name/Apache_beam_records/Test_records/',
      num_shards=20)

  with beam.Pipeline(options=pipeline_options) as pipe:
    _ = (pipe
         | 'ReadCSVFromText' >> read_textline_from_csv
         | 'LoadImageData' >> beam.ParDo(load_img_from_path)
         | 'PreprocessImages' >> beam.ParDo(augment_data)
         | 'ImageToTfExample' >> beam.ParDo(img_to_tfexample)
         | 'SerializeProto' >> beam.Map(lambda x: x.SerializeToString())
         | 'WriteTfRecord' >> write_to_tf_record)
    print('Done running')


def main():
  '''
  Main function to run the pipeline
  '''
  run_pipeline()


if __name__ == "__main__":
  logging.getLogger().setLevel(logging.INFO)
  PARSER = argparse.ArgumentParser()
  main()