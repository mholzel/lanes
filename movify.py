import cv2, os, tqdm
from PIL import Image


def convert(inputPath, outputPath, frameProcessor, fps=None):
    '''
    This function processes the video at the specified input path,
    saving the processed video at the specified output path.
    Specifically, each frame will be processed using the frame processor,
    and the output video will be saved at the specified fps.
    :param frameProcessor: a function which should take a frame as input,
    and returned the processed frame as output.
    If this function accepts more than one parameter, then the cumulative
    frame count will be passed as a second input.
    '''
    input_video = cv2.VideoCapture(inputPath)
    output_video = None
    count = 0

    # Set up the progress bar and grab the fps if one was not specified
    frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
    progressbar = tqdm.tqdm(total=frames)
    if fps is None:
        fps = input_video.get(cv2.CAP_PROP_FPS)

    # Now, for each frame of the video, run the frame processor and
    # save the output in a new video
    while input_video.isOpened():
        success, frame = input_video.read()
        if (not success) or (cv2.waitKey(10) & 0xFF == ord('q')):
            break
        if output_video is None:
            fourcc = 0x00000021
            output_video = cv2.VideoWriter(outputPath, fourcc, fps, (frame.shape[1], frame.shape[0]))
        count += 1
        try:
            out = frameProcessor(frame)
            output_video.write(out)
        except Exception as ex:
            print("Exception", ex, " when processing frame", count)
            output_video.write(frame)
        progressbar.update()
    progressbar.update(n=(frames - progressbar.n))
    progressbar.close()
    cv2.destroyAllWindows()
    input_video.release()
    if output_video is not None:
        output_video.release()


def save_video_frames(input_path, output_path):
    '''
    This function takes the input path to a video, and then saves all of the video frames to the specified output path.
    You will typically use this function if an algorithm is failing on a particular video frame, and you
    want to diagnose why. Specifically, in that case, you can save all of the video frames to file, and then
    repeatedly call your detection algorithm only on that single frame.

    Note: this function will create the output_path directory if it does not exist.
    '''

    # Make sure that the specified output directory already exists
    os.makedirs(output_path, exist_ok=True)

    # Open the video and save each of the frames
    input_video = cv2.VideoCapture(input_path)
    count = 0
    while input_video.isOpened():
        success, frame = input_video.read()
        if (not success) or (cv2.waitKey(10) & 0xFF == ord('q')):
            break
        count += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        Image.fromarray(frame).save(output_path + 'frame' + str(count) + '.jpg')
    cv2.destroyAllWindows()
    input_video.release()


if __name__ == "__main__":
    video = 'test_videos/aaa.mp4'
    save_video_frames(video, "aaa/")
