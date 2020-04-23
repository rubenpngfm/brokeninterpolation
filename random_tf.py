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
        write_frame(process2, process_frame(tf.convert_to_tensor(frame1, dtype=tf.uint16), tf.convert_to_tensor(frame2, dtype=tf.uint16)))
        write_frame(process2, frame2)

        
    process1.wait()
    process2.stdin.close()
    process2.wait()

def process_frame(frame1, frame2):
    '''Simple processing example: darken frame.'''
    r = tf.random.uniform(
        shape=[720,1280,3], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=None, name=None
    )
    return np.array(tf.math.add(tf.math.multiply(frame1, r, name=None), tf.math.multiply(frame2, tf.subtract(np.full((720,1280,3), 1, dtype=np.float32), r, name=None), name=None), name=None), dtype=np.uint8)
    

if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))
    t = time.time()
    run("demo.mp4", "demo_output.mp4", process_frame)
    print("Seconds: %s" % (time.time()-t))


