from moviepy.editor import VideoFileClip, clips_array, vfx

clip1 = VideoFileClip("test_videos/project_video.mp4")
clip2 = VideoFileClip("output_videos/project_video.mp4")
final_clip = clips_array([[clip1], [clip2]])
final_clip.write_videofile("output_videos/combined.mp4")
