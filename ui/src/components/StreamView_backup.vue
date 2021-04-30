<template>
<div>

    <div>
        <audio id="audio" autoplay="true"></audio>

        <video v-if="playing" style="height:200px" class="feature-image visible" id="myvideo" autoplay="true" playsinline="true"></video>
        <video v-else style="height:0px" class="invisible" id="myvideo" autoplay="true" playsinline="true"></video>
        <img v-if="!playing" :src="img">
    </div>

    <div class="mt-2 pb-2">
        <b-button size="sm" v-if="!playing" @click="start()" variant="success"><i class="fa fa-play"></i></b-button>
        <b-button size="sm" v-else @click="stop()" variant="danger"><i class="fa fa-stop"></i></b-button>
    </div>
</div>
</template>

<script>
// @ is an alias to /src
import axios from "axios";
import {
    appConfig,
    fetchSettings
} from "../app.config";

var time_start = null;

function current_stamp() {
    if (time_start === null) {
        time_start = new Date().getTime();
        return 0;
    } else {
        return new Date().getTime() - time_start;
    }
}

export default {
    props: {
        uid: String,
        height: String,
        img: String
    },
    data() {
        return {
            appConfig,
            streamUrl: "NA",
            searchQuery: "",
            items: {},
            selectedGroup: {},
            error: "",
            loadingData: true,
            message: "No Environments found.",
            pc: null,
            dc: null,
            dcInterval: null,
            video: {},
            playing: false
        };
    },
    mounted() {
        this.video = document.getElementById("myvideo");
    },
    methods: {
        selectGroup(groupId) {
            this.selectedGroup = this.$store.getters.envGroups[groupId];
            console.log("element: " + groupId);
        },

        createPeerConnection(config) {
            let pc = new RTCPeerConnection(config);

            pc.addEventListener("track", (evt) => {
                this.video.srcObject = evt.streams[0];
            });

            var parameters = {
                ordered: true
            };
            let dcInterval;

            let dc = pc.createDataChannel("chat", parameters);
            dc.onclose = function () {
                clearInterval(dcInterval);
                pc.close()
            };
            dc.onopen = function () {
                dcInterval = setInterval(function () {
                    var message = "ping " + current_stamp();
                    console.log(message);

                    dc.send(message);
                }, 1000);
            };
            dc.onmessage = function (evt) {
                if (evt.data.substring(0, 4) === "pong") {
                    var elapsed_ms =
                        current_stamp() - parseInt(evt.data.substring(5), 10);
                    console.log(" RTT " + elapsed_ms + " ms\n");
                }
            };
            return pc;
        },
        negotiate(pc) {
            pc.addTransceiver("video", {
                direction: "recvonly"
            });
            // pc.addTransceiver('audio', { direction: 'recvonly' });

            return pc
                .createOffer()
                .then(function (offer) {
                    return pc.setLocalDescription(offer);
                })
                .then(function () {
                    // wait for ICE gathering to complete
                    return new Promise(function (resolve, reject) {
                        if (pc.iceGatheringState === "complete") {
                            resolve();
                        } else {
                            var checkState = () => {
                                console.log("Checking state: " + pc.iceGatheringState);
                                if (pc.iceGatheringState === "complete") {
                                    pc.removeEventListener("icegatheringstatechange", checkState);
                                    resolve();
                                }
                            };

                            pc.addEventListener("icegatheringstatechange", checkState);
                        }
                    });
                })
                .then(() => {
                    var offer = pc.localDescription;
                    console.log(offer.toJSON());
                    return fetch(this.streamUrl, {
                        body: JSON.stringify({
                            stream_id: "SESSION_STREAM_" + this.uid,
                            sdp: offer.sdp,
                            type: offer.type,
                            video_transform: "none",
                        }),
                        headers: {
                            "Content-Type": "application/json",
                        },
                        method: "POST",
                    });
                })
                .then(function (response) {
                    return response.json();
                })
                .then(function (answer) {
                    console.log(JSON.stringify(answer, null, 2));
                    return pc.setRemoteDescription(answer);
                })
                .catch(function (e) {
                    alert(e);
                });
        },
        start() {
            this.playing = true
            var config = {
                sdpSemantics: "unified-plan",
                iceServers: [],
            };
            config.iceServers = [{
                urls: ["stun:stun.l.google.com:19302"]
            }];
            this.pc = this.createPeerConnection(config);
            this.negotiate(this.pc);

        },
        stop() {
            this.playing = false

            this.pc.close()
        }
    },
    computed: {
        filteredItems() {
            // item.info.name.toLowerCase().indexOf(this.search.toLowerCase())>=0
            console.log(this.searchQuery);
            const searchQuery = this.searchQuery;

            const filtered = {};
            Object.keys(this.$store.getters.envGroups).forEach((name) => {
                const match =
                    name.toLowerCase().indexOf(searchQuery.toLowerCase()) >= 0;
                if (match) {
                    filtered[name] = this.$store.getters.envGroups[name];
                }
            });
            return filtered;
        },
    },
    // Fetches posts when the component is created.
    created() {
        this.loadingData = true;
        fetchSettings().then(settings => {
            console.log(settings.sys_config.streamer_uri)
            this.streamUrl = settings.sys_config.streamer_uri
            console.log(this.streamUrl)
            // this.streamUrl = "http://localhost:" + settings['env']['STREAM_PORT']

        })

    },
    beforeDestroy() {
        if (this.pc)
            this.pc.close()

    }
};
</script>

<style>

</style>
