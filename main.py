import tensorflow as tf
import subprocess
import numpy as np
import ffmpeg
import time

def get_video_size(filename):
    probe = ffmpeg.probe(filename)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    return width, height

def start_ffmpeg_process1(in_filename):
    args = (
        ffmpeg
        .input(in_filename)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .compile()
    )
    return subprocess.Popen(args, stdout=subprocess.PIPE)


def start_ffmpeg_process2(out_filename, width, height):
    args = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
        .output(out_filename, pix_fmt='yuv420p')
        .overwrite_output()
        .compile()
    )
    return subprocess.Popen(args, stdin=subprocess.PIPE)

def read_frame(process1, width, height):
    # Note: RGB24 == 3 bytes per pixel.
    frame_size = width * height * 3
    in_bytes = process1.stdout.read(frame_size)
    if len(in_bytes) == 0:
        frame = None
    else:
        assert len(in_bytes) == frame_size
        frame = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([height, width, 3])
        )
    return frame

def write_frame(process2, frame):
    process2.stdin.write(
        frame
        .astype(np.uint8)
        .tobytes()
    )

def run(in_filename, out_filename, process_frame):
    width, height = get_video_size(in_filename)
    process1 = start_ffmpeg_process1(in_filename)
    process2 = start_ffmpeg_process2(out_filename, width, height)

    frame1 = read_frame(process1, width, height)
    frame2 = read_frame(process1, width, height)
    write_frame(process2, frame1)
    write_frame(process2, process_frame(frame1, frame2))
    write_frame(process2, frame2)
    while True:
        frame1 = frame2
        frame2 = read_frame(process1, width, height)
        if frame2 is None:
            break
        write_frame(process2, process_frame(frame1, frame2))
        #write_frame(process2, frame2)

        
    process1.wait()
    process2.stdin.close()
    process2.wait()

def process_frame(frame1, frame2):
    '''Simple processing example: darken frame.'''

    # simplify frame (round every RGB to the nearest 10) 0.19s input per frame



    sframe1 = tf.math.add(
        (frame1*0.1)*10, np.array(tf.math.greater(frame1, tf.math.add((frame1*0.1)*10, np.full((720,1280,3), 5), name=None), name=None))*10, name=None
    )



    # greater instead of greater_equal to avoid 255 -> 260 in uint8
    #sframe2 = tf.math.add(
    #    (frame2*0.1)*10, np.array(tf.math.greater(frame2, tf.math.add((frame2*0.1)*10, np.full((720,1280,3), 5), name=None), name=None))*10, name=None
    #)
    #sframe1 = np.array((np.array(frame1*0.05, dtype=np.uint8))*20, dtype=np.uint8)
    #sframe2 = np.array((np.array(frame2*0.05, dtype=np.uint8))*20, dtype=np.uint8)

    # frame differences - 0.20s input per frame
    dframe = tf.math.not_equal(
        frame1, frame2, name=None
    )
##    dframe, b, c = np.split(dframe, 3, axis=2)
##    dframe = dframe + b + c
##    dframe = tf.math.not_equal(
##        dframe, np.full((720, 1280, 1), True), name=None
##    )
##    np.stack(arrays, axis=2).shape



    # locate new pixel location
    #print(dframe.shape)
    return np.array((((frame1*0.5)+(frame2*0.5))*np.array(dframe)), dtype=np.uint8)

if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))
    t0 = time.time()
    run("demo.mp4", "demo_output.mp4", process_frame)
    print("Seconds: %f" % (time.time()-t0))


