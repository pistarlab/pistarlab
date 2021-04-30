import argparse
import asyncio
import fractions
import json
import logging
import pprint
import struct
import time
from typing import  Tuple

import aiohttp_cors

import numpy
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from av import AudioFrame, VideoFrame
from av.frame import Frame
from PIL import Image, ImageDraw

from pistarlab import ctx

# ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pc = None

icelist = []
AUDIO_PTIME = 0.020  # 20ms audio packetization
VIDEO_CLOCK_RATE = 90000
VIDEO_PTIME = 1 / 30  # 30fps
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)


class MediaStreamError(Exception):
    pass


class RedisVideoStreamTrack(MediaStreamTrack):

    kind = "video"

    _start: float
    _timestamp: int

    def __init__(self, stream_id):
        super().__init__()
        self.stream_id = stream_id
        self.sub = ctx.get_redis_client().pubsub()
        self.sub.subscribe(stream_id)
        self.default_img = Image.fromarray((numpy.zeros((200, 200, 3)) * 255).astype('uint8')).convert('RGB')

    async def next_timestamp(self) -> Tuple[int, fractions.Fraction]:
        if self.readyState != "live":
            raise MediaStreamError

        if hasattr(self, "_timestamp"):
            self._timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)
            wait = self._start + (self._timestamp / VIDEO_CLOCK_RATE) - time.time()
            await asyncio.sleep(wait)
        else:
            self._start = time.time()
            self._timestamp = 0
        return self._timestamp, VIDEO_TIME_BASE

    async def recv(self) -> Frame:
        """
        Receive the next :class:`~av.video.frame.VideoFrame`.

        The base implementation just reads a 640x480 green frame at 30fps,
        subclass :class:`VideoStreamTrack` to provide a useful implementation.
        """

        try:
            pts, time_base = await self.next_timestamp()

            encoded_data = 1
            while encoded_data == 1:
                encoded_data = self.sub.get_message()['data']
            size = struct.unpack('>II', encoded_data[:8])
            image_data = encoded_data[8:]
            img = Image.frombytes(mode="RGB", data=image_data, size=size)
            frame = VideoFrame.from_image(img)
            self.default_img = img.copy()
            # frame = VideoFrame.from_ndarray(np_array, format="bgr24")
            frame.pts = pts
            frame.time_base = time_base
        except:
            pts, time_base = await self.next_timestamp()
            img = self.default_img.copy()
            draw = ImageDraw.Draw(img)
            draw.text((5, 5), ".", (255, 255, 255))
            frame = VideoFrame.from_image(img)
            frame.pts = pts
            frame.time_base = time_base

        return frame

    def stop(self):
        self.sub.unsubscribe(self.stream_id)
        if not self.__ended:
            self.__ended = True
            self.emit("ended")

            # no more events will be emitted, so remove all event listeners
            # to facilitate garbage collection.
            self.remove_all_listeners()
            print("Stopping Stream {}".format(self.stream_id))


# Receive offer


async def offer(request):
    params = await request.json()
    stream_id = params['stream_id']
    logging.info(request.remote)
    sdp = params["sdp"]
    # sdp = re.sub('\S+\.local',request.remote,sdp)
    # sdp = re.sub('c=IN IP4 0\\.0\\.0\\.0',"c=IN IP4 {}".format(request.remote),sdp)
    logging.info(sdp)
    offer = RTCSessionDescription(sdp=sdp, type=params["type"])

    pc = RTCPeerConnection()

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info("ICE connection state is %s" % pc.iceConnectionState)
        if pc.iceConnectionState == "failed":
            await pc.close()

    @pc.on("track")
    def on_track(track):
        logger.info("Track %s received", track.kind)

        @track.on("ended")
        async def on_ended():
            logger.info("Track %s ended", track.kind)

    local_video = RedisVideoStreamTrack(stream_id)

    await pc.setRemoteDescription(offer)

    for t in pc.getTransceivers():
        if t.kind == "video" and local_video:
            pc.addTrack(local_video)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    logging.info("transceivers count {}".format(len(pc.getTransceivers())))

    logger.info(pprint.pformat(pc.localDescription.sdp))
    logger.info("---------------------")
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    if pc is not None:
        coros = pc.close()
        await asyncio.gather(*coros)

logging.basicConfig(level=logging.INFO)
app = web.Application()
app.on_shutdown.append(on_shutdown)
app.router.add_post("/offer", offer)

# Configure default CORS settings.
cors = aiohttp_cors.setup(app, defaults={
    "*": aiohttp_cors.ResourceOptions(
        allow_credentials=True,
        expose_headers="*",
        allow_headers="*",
    )
})

# Configure CORS on all routes.
for route in list(app.router.routes()):
    cors.add(route)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", help="port", default=7778)
    parser.add_argument("--host", help="host", default="0.0.0.0")
    args = parser.parse_args()
    ctx.connect()
    logging.info("Streamer is Ready")
    web.run_app(
        app,
        host=args.host,
        port=args.port,
        ssl_context=None,
        access_log=None)
