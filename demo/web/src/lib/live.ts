/** The webcam: one shared <video> element, shown by the viewer's preview and read by the live stream loop. */
let video: HTMLVideoElement | null = null;
let stream: MediaStream | null = null;

export function liveVideo(): HTMLVideoElement {
  if (!video) {
    video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    video.className = "block h-full w-full object-cover";
  }
  return video;
}

export async function openCamera(): Promise<HTMLVideoElement> {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("This browser cannot open a camera here. Use http://localhost (or HTTPS) in a recent browser.");
  }
  stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 960 }, height: { ideal: 720 }, facingMode: "environment" },
    audio: false,
  });
  const v = liveVideo();
  v.srcObject = stream;
  await v.play();
  return v;
}

export function closeCamera() {
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  if (video) video.srcObject = null;
}
