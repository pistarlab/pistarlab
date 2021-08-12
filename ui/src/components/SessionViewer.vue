<template>
<div>
    <h1><i class="fa fa-cube"></i> Session</h1>
    <b-modal id="def-modal" size="lg">
        <div>Config</div>
        <pre v-if="item.config">{{ JSON.parse(item.config) }}</pre>
        <div>Run Info</div>
        <pre v-if="item.runInfo">{{ JSON.parse(item.runInfo) }}</pre>
        <div>Summary</div>
        <pre v-if="item.summary">{{item.summary }}</pre>
    </b-modal>

    <b-button-toolbar>
        <b-button class="mr-2" variant="danger" v-if="item.status && item.status == 'RUNNING'" v-on:click="stopSession" size="sm"><i class="fa fa-stop"></i> Abort</b-button>

        <b-button class="mr-2" v-if="item && item.parentSession && item.parentSession.task" title="title" variant="secondary" :to="`/task/new/agenttask/${item.parentSession.task.ident}`" size="sm">
            <i class="fa fa-copy"></i> Duplicate
        </b-button>
        <b-button class="mr-2" v-else title="title" variant="secondary" :to="`/task/new/agenttask/${task.ident}`" size="sm">
            <i class="fa fa-copy"></i> Duplicate
        </b-button>

        <!-- <b-button title="Browse Data" class="mr-2" variant="secondary" :to="`/data_browser/?path=session/${uid}`" size="sm"><i class="fa fa-folder"></i> Browse Files</b-button> -->

        <b-button class="mr-2" title="Show Config" variant="secondary" v-b-modal="'def-modal'" size="sm"><i class="fa fa-cog"></i> View Configuration</b-button>
        <b-button v-if="item && !item.archived" variant="secondary" @click="updateArchive(true)" class="mr-2" size="sm"><i class="fa fa-eye"></i> Move to Archive</b-button>
        <b-button v-if="item && item.archived" variant="secondary" @click="updateArchive(false)" class="mr-2" size="sm"><i class="fa fa-eye"></i> Restore from Archive</b-button>
        <b-button-group class="ml-auto">
            <b-button size="sm" v-b-toggle.tasklogs variant="info">Task Log</b-button>
            <b-button size="sm" v-b-toggle.sessionlogs variant="info">Session Log</b-button>
        </b-button-group>
    </b-button-toolbar>
    <b-collapse id="tasklogs" class="mt-2" v-if="task">
        <LogViewer title="Task Log" :logStreamUrl="`${appConfig.API_URL}/api/stream/entity_logs/task/${task.ident}`"> </LogViewer>
    </b-collapse>

    <b-collapse id="sessionlogs" class="mt-2">
        <LogViewer title="Session Log" :logStreamUrl="`${appConfig.API_URL}/api/stream/entity_logs/session/${uid}`"> </LogViewer>
    </b-collapse>

    <div class="mt-5">
    </div>
    <b-container fluid>

        <b-row>
            <b-col cols=2 class="text-center">

                <b-alert v-if="message" show variant="warning">{{ message }}</b-alert>

                <div>
                    <div class="data_label">Session</div>
                    <h1>{{item.ident}} </h1>
                </div>
                <div class="mt-3">
                    <hr />
                </div>

                <div class="data_label">Parent</div>
                <div class="stat_value" v-if="parentSessionId">
                    <router-link :to="`/session/view/${parentSessionId}`">{{parentSessionId}}</router-link>

                </div>
                <div class="stat_value" v-else>
                    None
                </div>

                <div class="mt-3">
                </div>

                <div>
                    <div class="data_label">Task</div>
                    <span class="stat_value">
                        <router-link :to="`/task/view/${task.ident}`">{{
                    task.ident
                  }}</router-link>
                    </span>
                </div>
                <div class="mt-3">
                </div>

                <div>
                    <div class="data_label">State</div>
                    <span class="stat_value" v-if="item.status=='RUNNING'" style="color:green;font-weight:600">{{ item.status }}</span>
                    <span class="stat_value" v-else style="color:red;font-weight:600">{{ item.status }}</span>
                </div>
                <div class="mt-3">
                </div>

                <div>

                    <div class="data_label ">Created</div>
                    <span class="stat_value">{{ timedeltafordate(item.created) }} ago</span>
                </div>

                <div class="mt-3">
                </div>
                <div>
                    <div class="data_label">Runtime</div>
                    <span class="stat_value" v-if="item && item.summary">{{timelength(item.summary.runtime * 1000)}}</span>

                </div>
                <div class="mt-3">
                    <hr />
                </div>
                <div>
                    <h4>Agent</h4>
                    <AgentCardSmall :agent="item.agent"></AgentCardSmall>

                </div>
            </b-col>
            <b-col cols=10>
                <b-container fluid>
                    <b-row>
                        <b-col cols=4 class="text-center">
                            <div>
                                <div class="h4">
                                    <router-link :to="`/env_spec/view/${item.envSpecId }`">{{item.envSpec.displayedName }} </router-link>
                                </div>

                                <div>
                                    <img v-if="!playingLive && !playingEpisode" :src="`${appConfig.API_URL}/api/env_preview_image/${item.envSpec.environment.ident}`" alt="xxx" style="width:100%;" />
                                    <StreamView v-if="playingLive" :uid="uid" />
                                    <div v-if="playingLive" style="color:red;font-weight:900">Live</div>

                                    <video v-else-if="playingEpisode" loop autoplay controls style="width:100%">
                                        <source :src="videoURL" type="video/mp4">
                                    </video>

                                    <!-- <img class="feature-image" :src="imageURL" @error="imageError" height="300px" alt="No Preview Available" /> -->
                                </div>
                                <div class="mt-2">
                                    <b-button size="sm" v-if="!playingLive && liveAvailable" @click="startLive()" variant="success"><i class="fa fa-live"></i>Stream Live</b-button>

                                    <b-button size="sm" v-if="!playingEpisode && maxEpisode" @click="startEpisode()" variant="success"><i class="fa fa-play"></i> Episode {{maxEpisode}}</b-button>
                                    <span v-if="playingEpisode" class="data_label mr-1">

                                    </span>

                                    <b-button size="sm" v-if="playingEpisode" @click="stopPlaying()" variant="danger"><i class="fa fa-stop"></i> Episode {{ maxEpisode }}</b-button>
                                    <b-button size="sm" v-if="playingLive" @click="stopPlaying()" variant="danger"><i class="fa fa-stop"></i></b-button>
                                </div>

                                <div class="mt-2">
                                    <div v-if="maxEpisode">
                                        <router-link :to="`/episode/view/${item.ident}?episodeId=${maxEpisode}`">Total Recorded episodes: {{ totalRecordedEpisodes }}</router-link>
                                    </div>
                                    <div v-else-if="!loadingEpisodeData">
                                        No Episodes Recorded
                                    </div>
                                </div>
                            </div>
                            <div class="mt-4">
                                <hr />
                            </div>
                            <h4>Statistics</h4>
                            <b-container fluid>
                                <div v-if="item.summary">

                                    <b-row>

                                        <b-col>
                                            <div class="stat_label">Avg Reward/Episode</div>
                                            <span class="stat_value">{{
                          numberToString(formatNum(item.summary.mean_reward_per_episode, 4))
                        }}</span>
                                        </b-col>
                                        <b-col>

                                            <div class="stat_label">Avg Reward/Step</div>
                                            <span class="stat_value">{{
                          numberToString(formatNum(item.summary.mean_reward_per_step, 4))
                        }}</span>
                                        </b-col>
                                        <b-col>

                                            <div class="stat_label">
                                                Avg Reward/Step (Windowed)
                                            </div>
                                            <span class="stat_value">{{
                          numberToString(formatNum(item.summary.reward_mean_windowed, 4))
                        }}</span>
                                        </b-col>
                                    </b-row>
                                    <div class="mt-2">
                                    </div>
                                    <b-row>
                                        <b-col>
                                            <div class="stat_label">Steps/Episode</div>
                                            <span class="stat_value">{{
                          numberToString(formatNum(item.summary.mean_steps_per_episode, 4))
                        }}</span>
                                        </b-col>
                                        <b-col>
                                            <div class="stat_label">Total Episodes</div>
                                            <span class="stat_value">{{
                          numberToString(item.summary.episode_count)
                        }}</span>
                                        </b-col>
                                        <b-col>
                                            <div class="stat_label">Total Reward</div>
                                            <span class="stat_value">{{
                          numberToString(formatNum(item.summary.reward_total,8))
                        }}</span>
                                        </b-col>
                                    </b-row>
                                    <div class="mt-2">
                                    </div>
                                    <b-row>
                                        <b-col>
                                            <div class="stat_label">Total Steps</div>
                                            <span class="stat_value">{{
                          numberToString(item.summary.step_count)
                        }}</span>
                                        </b-col>
                                        <b-col>
                                            <div class="stat_label">Steps/Second</div>
                                            <span class="stat_value">{{
                          numberToString(formatNum(item.summary.steps_per_second, 4))
                        }}</span>
                                        </b-col>
                                        <b-col>

                                            <div class="stat_label">Best Episode Reward Total</div>
                                            <span class="stat_value">{{numberToString(formatNum(item.summary.best_ep_reward_total, 4))}}</span>
                                        </b-col>
                                    </b-row>
                                    <div class="mt-2">
                                    </div>
                                    <b-row>
                                        <b-col>
                                            <div class="stat_label">Best Episode Reward Mean over Window</div>
                                            <span class="stat_value">{{numberToString(formatNum(item.summary.best_ep_reward_mean_windowed, 4))}}</span>
                                        </b-col>
                                    </b-row>

                                </div>
                                <div v-else>
                                    <b-row>
                                        <b-col>
                                            Summary data is missing
                                        </b-col>
                                    </b-row>
                                </div>
                            </b-container>
                        </b-col>
                        <b-col cols=8>
                            <b-container fluid>
                                <b-row>
                                    <b-col v-for="(graph,i) in graphListResults" :key="i" cols=6>
                                        <div v-if="graph && graph.graphData" class="mb-4">

                                            <div v-if="graph.graphData.chartData">
                                                <!-- <PlotlyVue :data="graph.graphData.data" :layout="graph.graphData.layout" :display-mode-bar="false"></PlotlyVue> -->
                                                <LineChart :chart-data="graph.graphData.chartData" :options="graph.graphData.chartOptions" :key="graphKey"></LineChart>

                                            </div>

                                        </div>

                                    </b-col>

                                    <b-button class="mr-2" v-on:click="graphKey++" size="sm">reset view</b-button>
                                </b-row>
                            </b-container>
                        </b-col>
                    </b-row>
                    <b-row>

                    </b-row>
                </b-container>

            </b-col>
            <b-col cols=2>

            </b-col>
        </b-row>

    </b-container>

    <div class="mt-4"></div>

</div>
</template>

<script>
// @ is an alias to /src
import axios from "axios";
import {
    appConfig
} from "../app.config";
import gql from "graphql-tag";
import {
    timelength,
    timedelta,
    timepretty,
    numberToString,
    timedeltafordate,
    formatNum
} from "../funcs";
import AgentCardSmall from "../components/AgentCardSmall.vue";
import LogViewer from "../components/LogViewer.vue";

import StreamView from "../components/StreamView.vue";
// import {
//     Plotly as PlotlyVue
// } from 'vue-plotly'
import LineChart from "../components/LineChart.vue";

const sessionConfigFields = [{
    key: "item.summary.episode_counter",
    label: "Episodes",
}, ];

export default {
    name: "Session",
    components: {
        StreamView,
        LineChart,
        AgentCardSmall,
        LogViewer
    },
    apollo: {
        // Simple query that will update the 'hello' vue property

    },
    data() {
        return {
            episodes: [],
            totalRecordedEpisodes: null,
            maxEpisode: null,
            activeGraphs: {},
            appConfig,
            loadingEpisodeData: true,
            liveAvailable: false,
            playingEpisode: false,
            playingLive: false,

            imageURL: "placeholder.jpg",
            videoURL: "",
            graphListResults: [],
            graphKey: 0,
            graphList: [{
                    title: "Reward per Episode",
                    key: "episode_reward_total",
                    group: "ep_stats",
                    value: "episode_reward_total",
                    stepField: "episode_count",
                    color: 'rgba(0, 255, 0,1)',
                    colormin: 'rgba(0, 255, 0,0.2)',
                    colormax: 'rgba(0, 255, 0,0.2)',
                    stats: [{}],
                    binSize: 20,
                    count: 0,
                },

                {
                    title: "Steps per Episode",
                    key: "episode_step_count",
                    group: "ep_stats",
                    value: "episode_step_count",
                    stepField: "episode_count",
                    color: 'rgb(255, 160, 0)',
                    colormin: 'rgba(255, 160, 0,0.2)',
                    colormax: 'rgba(255, 160, 0,0.2)',
                    stats: [{}],
                    binSize: 20,
                    count: 0,
                },

                {
                    title: "Reward Total per Step",
                    key: "reward_total",
                    group: "step_stats",
                    value: "reward_total",
                    stepField: "step_count",
                    color: 'rgb(255, 100, 100)',
                    colormin: 'rgba(255, 100, 100,0.2)',
                    colormax: 'rgba(255, 100, 100,0.2)',
                    stats: [{}],
                    binSize: 20,
                    count: 0,
                },
                {
                    title: "Step Latency",
                    key: "step_latency_mean_windowed",
                    group: "step_stats",
                    value: "step_latency_mean_windowed",
                    stepField: "step_count",
                    color: 'rgb(55, 128, 191)',
                    colormin: 'rgba(55, 128, 191,0.2)',
                    colormax: 'rgba(55, 128, 191,0.2)',
                    stats: [{}],
                    binSize: 20,
                    count: 0,
                },
            ],
            timer: "",
            timer2: "",
            logtxt: "",
            message: "",
            error: "",
            testData: {},
            autoRefresh: true,
            es: null,
            plots: {},
        };
    },
    props: {
        uid: String,
        session: Object
    },
    computed: {
        item() {
            if (this.session) return this.session
            else return {
                'agent': {}
            }
        },
        task() {
            if (this.session && this.session.task) {
                return this.session.task
            } else
                return {}
        },
        parentSessionId() {

            if (this.session && this.session.parentSessionId) {
                return this.session.parentSessionId
            } else
                return null
        }
    },
    mounted() {
        //
    },
    methods: {

        updateArchive(archive) {
            this.$apollo.mutate({
                mutation: gql `mutation archiveSessionMutation($id:String!,$archive:Boolean) 
                {
                    sessionSetArchive(id:$id, archive:$archive){
                        success
                        }
                }`,
                variables: {
                    id: this.session.id,
                    archive: archive
                },
            }).then((data) => {
                this.$emit('update')
            }).catch((error) => {
                console.error(error)
            })
        },
        timedelta,
        timelength,
        timedeltafordate,
        numberToString,
        formatNum,
        getImageId(uid) {
            if (uid) {

                let id = parseInt(uid.split("-")[1]);
                return id % 19;
            } else ""
        },
        startLive() {
            this.playingLive = true
            this.playingEpisode = false

        },
        startEpisode() {
            this.playingEpisode = true
            this.playingLive = false
        },
        stopPlaying() {
            this.playingEpisode = false
            this.playingLive = false
        },

        imageError(event) {
            console.log(event);
            this.imageURL = "placeholder.jpg";
        },
        refreshData() {
            this.liveAvailable = (this.item.status == "RUNNING")
            if (this.item.status == null || (this.item.status && this.item.status == "RUNNING")) {
                this.loadData()
                this.loadGraphs()
            } else {
                clearInterval(this.timer);
                return
            }
        },

        loadData() {
            this.loadingEpisodeData = true

            axios
                .get(`${appConfig.API_URL}/api/session_max_episode_recorded/${this.uid}`)
                .then((response) => {
                    if (response.data["max_recorded_ep"] && response.data["max_recorded_ep"] != "undefined") {
                        this.maxEpisode = response.data["max_recorded_ep"];

                        this.totalRecordedEpisodes = response.data["total_recorded"];
                        this.imageURL = `${appConfig.API_URL}/api/session_episode_gif/${this.uid}/${this.maxEpisode}`;
                        this.videoURL = `${appConfig.API_URL}/api/session_episode_mp4/${this.uid}/${this.maxEpisode}`;
                        console.log("RESULTS ARRIVED")
                        this.loadingEpisodeData = false
                    }
                })
                .catch((e) => {
                    this.error = e;
                    this.loadingEpisodeData = false
                });

        },
        loadGraphs() {
            // this.graphListResults = []

            this.graphList.forEach((graphItem, idx) => {
                let url = `${appConfig.API_URL}/api/session_plots_json/${this.uid}/${graphItem.group}/${graphItem.value}/${graphItem.stepField}?bin_size=${graphItem.binSize}`;
                // console.log("Fetching: "+ url);

                axios
                    .get(
                        url
                    )
                    .then((response) => {
                        const graph = response.data;

                        if (graph.data == undefined) {
                            return
                        }
                        var chartData = null;
                        if (graph.include_stats) {

                            chartData = {
                                labels: graph.data.idxs,

                                datasets: [{
                                        label: graphItem.key,
                                        backgroundColor: graphItem.color,
                                        borderColor: graphItem.color,
                                        data: graph.data.means,
                                        fill: false
                                    },
                                    {
                                        label: graphItem.key,
                                        backgroundColor: graphItem.colormax,
                                        data: graph.data.uppers,
                                        fill: '-1',
                                        borderColor: 'rgba(0,0,0,0);',
                                        borderWidth: 0,
                                        pointRadius: 0,
                                        spanGaps: false

                                    },
                                    {
                                        label: graphItem.key,
                                        backgroundColor: graphItem.colormin,
                                        data: graph.data.lowers,
                                        fill: '-2',
                                        borderColor: 'rgba(0,0,0,0);',

                                        pointRadius: 0,
                                        borderWidth: 0,
                                        spanGaps: false,

                                    }
                                ]
                            }
                        } else {

                            chartData = {
                                labels: graph.data.idxs,

                                datasets: [{
                                    label: graphItem.key,
                                    backgroundColor: graphItem.color,
                                    borderColor: graphItem.color,
                                    data: graph.data.vals,
                                    fill: false
                                }]
                            }

                        }
                        graphItem.graphData = {}
                        graphItem.graphData.chartData = chartData

                        graphItem.graphData.chartOptions = {
                            responsive: true,
                            // fill: false,
                            animation: false,
                            maintainAspectRatio: false,
                            title: {
                                display: true,
                                text: graphItem.title
                            },
                            legend: {
                                display: false
                            },
                            pan: {
                                enabled: true,
                                mode: 'xy'
                            },
                            zoom: {
                                // Boolean to enable zooming
                                enabled: true,

                                // Zooming directions. Remove the appropriate direction to disable 
                                // Eg. 'y' would only allow zooming in the y direction
                                mode: 'xy',
                            },
                            // animations: {
                            //     tension: {
                            //         duration: 200,
                            //         easing: 'linear',
                            //         from: 1,
                            //         to: 0,
                            //         loop: true
                            //     }
                            // }
                        }

                        // this.graphListResults.push(graphItem)
                        this.$set(this.graphListResults, idx, graphItem)

                    })
                    .catch((e) => {
                        console.log(e);
                        this.error = e;
                    });
            });
        },
        stopSession() {
            if (this.item) {
                axios
                    .get(
                        `${appConfig.API_URL}/api/admin/task/stop/${this.item.task.ident}`
                    )
                    .then((response) => {
                        let message = response.data["message"];
                        this.makeToast(message, "User Abort Request", "info")
                        console.log(` ${message}`)
                        this.refreshData()
                    })
                    .catch((e) => {
                        this.error = e;
                        this.message = this.error;
                    });
            }
        },
    },
    created() {
        console.log(this.uid);
        this.loadData();
        this.loadGraphs();
        this.timer = setInterval(this.refreshData, 2000);
    },
    beforeDestroy() {
        if (this.es) {
            this.es.close();
        }
        clearInterval(this.timer);
    },
};
</script>

<style>
/* a.page-link{
   color: black;
 } */
.plot {
    height: 350px;
}

.plotcontainer {
    border-style: solid;
    border-width: 1px;
    border-color: #ccc;
    /* padding: 10px; */
    margin: 4px;
}
</style>
