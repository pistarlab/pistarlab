<template>
<div>
    <h1><i class="fa fa-cubes"></i> Multi-Agent Session: <span v-if="item">{{item.ident}}</span></h1>

    <b-modal id="def-modal" size="lg">
        <div>Config</div>
        <pre v-if="item.config">{{ JSON.parse(item.config) }}</pre>
        <div>Run Info</div>
        <pre v-if="item.config">{{ JSON.parse(item.runInfo) }}</pre>

    </b-modal>
    <b-button-toolbar>
        <b-button-group class="mr-auto">
            <b-button class="mr-2" title="title" variant="secondary" :to="`/task/new/agenttask/${task.ident}`" size="sm">
                <i class="fa fa-copy"></i> Copy Task
            </b-button>
            <b-button class="mr-2" title="Show Config" variant="secondary" v-b-modal="'def-modal'" size="sm"><i class="fa fa-info-circle"></i> View Configuration</b-button>
            <SessionRuntimeController :item="item"></SessionRuntimeController>

        </b-button-group>
        <b-button-group class="ml-auto">

            <b-button size="sm" v-b-toggle.tasklogs variant="info">Task Log</b-button>

            <b-button size="sm" v-b-toggle.sessionlogs variant="info">Session Log</b-button>
        </b-button-group>

    </b-button-toolbar>
    <div class="mt-4"></div>

    <div class="mt-4"></div>
    <b-collapse id="tasklogs" class="mt-2" v-if="task">
        <LogViewer title="Task Log" :logStreamUrl="`${appConfig.API_URL}/api/stream/entity_logs/task/${task.ident}`"> </LogViewer>
    </b-collapse>

    <b-collapse id="sessionlogs" class="mt-2">
        <LogViewer title="Session Log" :logStreamUrl="`${appConfig.API_URL}/api/stream/entity_logs/session/${uid}`"> </LogViewer>
    </b-collapse>
    <b-container fluid>

        <b-row>
            <b-col>
                <b-alert v-if="message" show variant="warning">{{ message }}</b-alert>
            </b-col>
        </b-row>

        <div class="mt-4"></div>

        <b-row class="text-center">
            <b-col cols=2>
                <div>
                    <div class="data_label">Session Id</div>
                    <h1>{{item.ident}}
                    </h1>
                </div>
                <div class="mt-3">
                </div>
                <div>
                    <div class="data_label">Task Id</div>
                    <span>
                        <router-link :to="`/task/view/${task.ident}`">{{
                    task.ident
                  }}</router-link>
                    </span>
                </div>
                <div class="mt-3">
                </div>
                <div>
                    <div class="data_label">State</div>
                    <span>{{ item.status }}</span>
                </div>
                <div class="mt-3">
                </div>
                <div>

                    <div class="data_label ">Creation Time</div>
                    <span>{{ item.created}}</span>
                </div>
                <div class="mt-3">
                </div>
                <div>
                    <div class="data_label">Runtime (seconds)</div>
                    <span v-if="item && item.summary">{{timelength(formatNum(item.summary.runtime,4) * 1000)}}</span>
                                        

                </div>
            </b-col>

            <b-col cols=10>
                <b-container fluid>
                    <b-row>
                       
                        <!-- liveAvailable:{{liveAvailable}},playingLive:{{playingLive}},playingEpisode:{{playingEpisode}} -->
                        <b-col cols=4 class="text-center">
                            <div>
                                <div class="text-center">
                                    <div class="mb-2">
                                        <router-link class="h3" :to="`/env_spec/view/${item.envSpecId }`"> {{item.envSpec.displayedName }}</router-link>
                                    </div>
                                    <div>
                                        <img v-if="!playingLive && !playingEpisode" :src="`${appConfig.API_URL}/api/env_preview_image/${item.envSpecId}`" alt="" style="height:100%;max-width:400px" />
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
                            </div>

                        </b-col>

                        <b-col class="d-flex justify-content-around">
                            <div>
                                <h3> Agent Sessions</h3>

                                <div v-if="Object.keys(childSessions).length > 0">
                                    <b-card>

                                        <b-table striped hover table-busy :items="childSessions" :fields="fields">
                                            <template v-slot:cell(ident)="data">
                                                <!-- `data.value` is the value after formatted by the Formatter -->
                                                
                                                
                                                <router-link :to="`/session/view/${data.item.ident}`">{{ data.item.ident }}</router-link>
                                            </template>

                                            <template v-slot:cell(agentId)="data">
                                                <!-- `data.value` is the value after formatted by the Formatter -->
                                                <span v-if="data.item.agent">
                                                    <b-img  style="max-height:30px;" :src="`/img/agent_spec_icons/agent_${getImageId(data.item.agent.specId)}.png`" alt="Image" class="rounded-0 mr-4" ></b-img> 
                                                    <router-link :to="`/agent/view/${data.item.agent.ident}`"><span v-if="data.item.agent && data.item.agent.name">{{data.item.agent.name}}</span><span v-else>{{data.item.agent.ident}}</span></router-link> ({{ data.item.agent.specId }})
                                                </span>
                                            </template>
                                            <template v-slot:cell(taskId)="data">
                                                <!-- `data.value` is the value after formatted by the Formatter -->
                                                <router-link :to="`/task/view/${data.item.task.ident}`">{{ data.item.task.ident }}</router-link>
                                            </template>
                                        </b-table>
                                    </b-card>
                                    <p>{{ error }}</p>
                                </div>

                            </div>
                        </b-col>
                    </b-row>
                </b-container>

            </b-col>
        </b-row>
    </b-container>

    <b-container fluid>

        <h4>Statistics</h4>
        <div class="mt-3"></div>
        <div v-if="item.summary">
            <b-row>
                <b-col>

                    <div>
                        <div class="stat_label">Avg Reward/Episode</div>
                        <span class="stat_value">{{
                          formatNum(item.summary.mean_reward_per_episode, 4)
                        }}</span>
                    </div>
                </b-col>
                <b-col>
                    <div>
                        <div class="stat_label">Avg Reward/Step</div>
                        <span class="stat_value">{{
                          formatNum(item.summary.mean_reward_per_step, 4)
                        }}</span>
                    </div>
                </b-col>
                <b-col>
                    <div>
                        <div class="stat_label">
                            Avg Reward/Step (Windowed)
                        </div>
                        <span class="stat_value">{{
                          formatNum(item.summary.reward_mean_windowed, 4)
                        }}</span>
                    </div>
                </b-col>
                <b-col>
                    <div>
                        <div class="stat_label">Steps/Episode</div>
                        <span class="stat_value">{{
                          formatNum(item.summary.mean_steps_per_episode, 4)
                        }}</span>
                    </div>
                </b-col>

                <b-col>

                    <div>
                        <div class="stat_label">Completed Episodes</div>
                        <span class="stat_value">{{
                          item.summary.episode_count
                        }}</span>
                    </div>
                </b-col>
                <b-col>
                    <div>
                        <div class="stat_label">Total Reward</div>
                        <span class="stat_value">{{
                          formatNum(item.summary.reward_total)
                        }}</span>
                    </div>
                </b-col>

                <b-col>
                    <div>
                        <div class="stat_label">Total Steps</div>
                        <span class="stat_value">{{
                          item.summary.step_count
                        }}</span>
                    </div>

                </b-col>
                <b-col>

                    <div>
                        <div class="stat_label">Steps/Second</div>
                        <span class="stat_value">{{
                          formatNum(item.summary.steps_per_second, 4)
                        }}</span>
                    </div>

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
        <div class="mt-4"></div>
    </b-container>
    <div class="mt-4"></div>
    <hr/>
    <b-container fluid class="mt-4">

        <b-row>
            <b-col cols=3 v-for="graph in graphList" :key="graph.key">

                <div v-if="graph.graphData">

                    <div v-if="graph.graphData.data && (graph.graphData.data.length > 0 ) && graph.graphData.data[0] && graph.graphData.data[0].x.length > 0">
                        <PlotlyVue :data="graph.graphData.data" :layout="graph.graphData.layout" :display-mode-bar="false"></PlotlyVue>
                    </div>

                </div>

            </b-col>
        </b-row>
    </b-container>

</div>
</template>

<script>
// @ is an alias to /src
import axios from "axios";
import LogViewer from "../components/LogViewer.vue";

import {
    appConfig
} from "../app.config";
import gql from "graphql-tag";
import {
    timelength,
    timedelta,
    timepretty
} from "../funcs";
import SessionRuntimeController from "../components/SessionRuntimeController.vue";

import StreamView from "../components/StreamView.vue";
import {
    Plotly as PlotlyVue
} from 'vue-plotly'

const sessionConfigFields = [{
    key: "item.summary.episode_counter",
    label: "Episodes",
}, ];

const fields = [{
        key: "ident",
        label: "Session Id",
    },
    {
        key: "agentId",
        label: "Agent",
    },
    {
        key: "summary.step_count",
        label: "Total Steps",
    },
    {
        key: "summary.episode_count",
        label: "Episode Count",
    }
];

export default {
    name: "Session",
    components: {
        StreamView,
        PlotlyVue,
        LogViewer,
        SessionRuntimeController
    },
    apollo: {
        //
    },
    data() {
        return {
            graph: "",
            episodes: [],
            totalRecordedEpisodes: null,
            maxEpisode: "",
            playingPreview: false,
            loadingEpisodeData: false,
            playingLive: false,
            playingEpisode: false,
            activeGraphs: {},
            fields,
            appConfig,
            imageURL: "placeholder.jpg",
            videoURL: "",

            // sessionsGraphList: [{
            //         key: "episode_reward_total",
            //         group: "ep_stats",
            //         value: "episode_reward_total",
            //         stepField: "episode_count",
            //         title: "Total Reward per Episode",
            //         stats: [{}]
            //     },

            //     {
            //         key: "episode_step_count",
            //         group: "ep_stats",
            //         value: "episode_step_count",
            //         stepField: "episode_count",
            //         stats: [{}]
            //     },

            //     {
            //         key: "reward_total",
            //         group: "step_stats",
            //         value: "reward_total",
            //         stepField: "step_count",
            //         stats: [{}]
            //     },
            //     {
            //         key: "step_latency_mean_windowed",
            //         group: "step_stats",
            //         value: "step_latency_mean_windowed",
            //         stepField: "step_count",
            //         stats: [{}]
            //     },
            // ],
            graphList: [{
                    key: "episode_reward_total",
                    group: "ep_stats",
                    value: "episode_reward_total",
                    stepField: "episode_count",
                    title: "Total Reward per Episode",
                    stats: [{}]
                },

                {
                    key: "episode_step_count",
                    group: "ep_stats",
                    value: "episode_step_count",
                    stepField: "episode_count",
                    stats: [{}]
                },

                {
                    key: "reward_total",
                    group: "step_stats",
                    value: "reward_total",
                    stepField: "step_count",
                    stats: [{}]
                },
                {
                    key: "step_latency_mean_windowed",
                    group: "step_stats",
                    value: "step_latency_mean_windowed",
                    stepField: "step_count",
                    stats: [{}]
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
            plots: {}
        };
    },
    props: {
        uid: String,
        session: Object
    },
    computed: {
        liveAvailable(){

            return this.item != null && this.item.status == "RUNNING";

        },
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
        childSessions() {
            if (this.session && this.session.childSessions) {
                return this.session.childSessions.edges.map(n => n.node)

            } else
                return []
        },
        uids() {
            if (this.childSessions.length > 0) {
                let uids = this.childSessions.map((n) => n.ident)
                return uids

            } else
                return []
        }

    },
    watch: {
        childSessions: function (val) {
            this.loadGraphs()
        }

    },
    mounted() {
        //
    },
    methods: {
        timepretty,
        timedelta,
        timelength,
        formatNum(num, prec) {
            if (num == null) {
                return "";
            } else if (isNaN(num)) {
                return "NaN";
            } else {
                return num.toPrecision(prec);
            }
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
                        this.loadingEpisodeData = false
                    }
                })
                .catch((e) => {
                    this.error = e;
                    this.loadingEpisodeData = false
                });

        },
        loadGraphs() {
            this.graphList.forEach((graphItem, idx) => {
                Promise.all(this.uids.map((uid) => {
                    return axios
                        .get(
                            `${appConfig.API_URL}/api/session_plotly_json/${uid}/${graphItem.group}/${graphItem.value}/${graphItem.stepField}`
                        )
                        .then((response) => {
                            return response.data;
                        })
                        .catch((e) => {
                            console.log(e);
                            this.error = e;
                        });
                })).then((dataList) => {

                    const graph = dataList[0];
                    if (!graph || typeof graph.layout === 'undefined') return

                    const graphData = {};
                    graphData.data = []

                    dataList.forEach(d => {
                        if (d.data) {
                            graphData.data.push(d.data)
                        }
                    })
                    graphData.layout = graph.layout;
                    graphData.layout.autosize = true
                    graphData.layout.margin = {
                        t: 25,
                        l: 25,
                        r: 25,
                        b: 25
                    };
                    graphData.layout.font = {
                        color: "rgba(200,200,200,1)"
                    }
                    graphData.layout.plot_bgcolor = "rgba(0,0,0,0)";
                    graphData.layout.paper_bgcolor = "rgba(0,0,0,0)";
                    graphData.layout.yaxis = {

                        "gridcolor": "rgba(200,200,200,0.25)",
                        "gridwidth": 1,

                    }
                    graphData.layout.xaxis = {

                        "gridcolor": "rgba(200,200,200,0.25)",
                        "gridwidth": 1,

                    }
                    graphData.layout.height = 250;
                    graphData.config = graphItem;
                    graphItem.graphData = graphData;
                    this.$set(this.graphList, idx, graphItem);

                })
            });
        },

        stopSession() {
            if (this.item) {
                axios
                    .get(
                        `${appConfig.API_URL}/api/admin/task/stop/${this.item.task.ident}`
                    )
                    .then((response) => {
                        // JSON responses are automatically parsed.

                        this.message = response.data["message"];
                        console.log(`TASK ABORT REQUEST ${this.message}`)
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
    background-color: #000;
}

.plotcontainer {
    border-style: solid;
    border-width: 1px;
    border-color: #ccc;
    /* padding: 10px; */
    margin: 4px;
}
</style>
