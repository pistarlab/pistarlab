<template>
<div>
    <h1><i class="fa fa-cubes"></i> Multi Agent Session</h1>
    <hr />

    <b-modal id="def-modal" size="lg">
        <div>Config</div>
        <pre v-if="item.config">{{ JSON.parse(item.config) }}</pre>
        <div>Run Info</div>
        <pre v-if="item.config">{{ JSON.parse(item.runInfo) }}</pre>

    </b-modal>
     <b-button-toolbar>
    <b-button-group class="mr-auto">
        <b-button class="mr-2" variant="danger" v-if="item.status && item.status == 'RUNNING'" v-on:click="stopSession" size="sm">Abort Task</b-button>
        <b-button class="mr-2" title="title" variant="secondary" :to="`/task/new/agenttask/${task.ident}`" size="sm">
            <i class="fa fa-copy"></i> Duplicate Session
        </b-button>
        <b-button class="mr-2" title="Show Config" variant="secondary" v-b-modal="'def-modal'" size="sm"><i class="fa fa-cog"></i> View Configuration</b-button>
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
        <b-card>

            <b-row>
                <b-col>
                    <b-alert v-if="message" show variant="warning">{{ message }}</b-alert>
                </b-col>
            </b-row>
            <b-row>
                <b-col>
                    <div class="data_label">Session Id</div>
                    <span>{{item.ident}}
                    </span>
                </b-col>

                <b-col>
                    <div class="data_label">Task Id</div>
                    <span>
                        <router-link :to="`/task/view/${task.ident}`">{{
                    task.ident
                  }}</router-link>
                    </span>
                </b-col>

                <b-col>
                    <div class="data_label">State</div>
                    <span>{{ item.status }}</span>
                </b-col>

                <b-col>

                    <div class="data_label ">Creation Time</div>
                    <span>{{ item.created }}</span>
                </b-col>

                <b-col>
                    <div class="data_label">Runtime (seconds)</div>
                    <span v-if="item && item.summary">{{
                                        timelength(
                                            formatNum(
                                            item.summary.runtime,
                                            4
                                            )
                                        )
                                        }}</span>

                </b-col>

                <!-- <b-col>
                <div class="data_label">Comment</div>
                <span v-if="item.comments">{{ item.comments }}</span>
                <span v-else>+</span>
            </b-col> -->

            </b-row>
        </b-card>
        <div class="mt-4"></div>

        <b-row class="text-center">

            <b-col class="d-flex justify-content-around">
                <b-card>
                    <div>
                        <h3>Environment</h3>

                        <div class="text-center" style="height:320px;">
                            <div>
                                <router-link :to="`/env_spec/view/${item.envSpecId }`"> {{item.envSpecId }}</router-link>
                            </div>
                            <div class="mt-2">
                                <div v-if="!playingPreview">
                                    <img v-if="item.envSpec && item.envSpec.environment && item.envSpec.environment.ident" :src="`${appConfig.API_URL}/api/env_preview_image/${item.envSpec.environment.ident}`" alt="" style="height:260px;" />
                                </div>

                                <div v-else>
                                    <div v-if="item.status && item.status == 'RUNNING'" height="300px">
                                        <StreamView :uid="uid" />
                                    </div>
                                    <div v-else-if="maxEpisode">

                                        <embed :src="videoURL" type="video/mp4" style="width: 100%;height: 100%;">

                                        <div class="data_label">
                                            Showing Episode: {{ maxEpisode }}
                                        </div>
                                        <!-- <img class="feature-image" :src="imageURL" @error="imageError" height="300px" alt="No Preview Available" /> -->
                                    </div>
                                </div>
                                <div>
                                    <div v-if="maxEpisode">
                                        <router-link :to="`/episode/view/${item.ident}?episodeId=${maxEpisode}`">Total Recorded episodes: {{ totalRecordedEpisodes }}</router-link>
                                    </div>
                                    <div v-else>
                                        No Episodes Recorded
                                    </div>

                                </div>
                            </div>
                        </div>
                    </div>
                </b-card>

            </b-col>

            <b-col class="d-flex justify-content-around">
                <b-card>
                    <div>
                        <h3> Player Sessions</h3>

                        <div v-if="Object.keys(childSessions).length > 0">
                            <b-table striped hover table-busy :items="childSessions" :fields="fields">
                                <template v-slot:cell(ident)="data">
                                    <!-- `data.value` is the value after formatted by the Formatter -->
                                    <router-link :to="`/session/view/${data.item.ident}`">{{ data.item.ident }}</router-link>
                                </template>

                                <template v-slot:cell(agentId)="data">
                                    <!-- `data.value` is the value after formatted by the Formatter -->
                                    <span v-if="data.item.agent">
                                        <router-link :to="`/agent/view/${data.item.agent.ident}`">{{ data.item.agent.ident }}</router-link> ({{ data.item.agent.specId }})
                                    </span>
                                </template>
                                <template v-slot:cell(taskId)="data">
                                    <!-- `data.value` is the value after formatted by the Formatter -->
                                    <router-link :to="`/task/view/${data.item.task.ident}`">{{ data.item.task.ident }}</router-link>
                                </template>
                            </b-table>
                            <p>{{ error }}</p>
                        </div>

                    </div>
                </b-card>
            </b-col>
        </b-row>
    </b-container>

    <b-card title="Statistics">
        <b-container fluid>
            <h3></h3>
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
                            <div class="stat_label">Total Episodes</div>
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

            <div class="mt-4"></div>

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
    </b-card>

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
        LogViewer
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
        timedelta,
        timelength,
        getImageId(uid) {
            if (uid) {

                let id = parseInt(uid.split("-")[1]);
                return id % 19;
            } else ""
        },
        formatNum(num, prec) {
            if (num == null) {
                return "";
            } else if (isNaN(num)) {
                return "NaN";
            } else {
                return num.toPrecision(prec);
            }
        },
        imageError(event) {
            console.log(event);
            this.imageURL = "placeholder.jpg";
        },
        refreshData() {
            if (this.item.status == null || (this.item.status && this.item.status == "RUNNING")) {
                // this.$apollo.queries.session.refetch();
                this.loadData()
                this.loadGraphs()
            } else {
                clearInterval(this.timer);
                return
            }
        },

        loadData() {

            axios
                .get(`${appConfig.API_URL}/api/session_max_episode_recorded/${this.uid}`)
                .then((response) => {
                    if (response.data["max_recorded_ep"] && response.data["max_recorded_ep"] != "undefined") {
                        this.maxEpisode = response.data["max_recorded_ep"];

                        this.totalRecordedEpisodes = response.data["total_recorded"];
                        this.imageURL = `${appConfig.API_URL}/api/session_episode_gif/${this.uid}/${this.maxEpisode}`;
                        this.videoURL = `${appConfig.API_URL}/api/session_episode_mp4/${this.uid}/${this.maxEpisode}`;
                    }
                })
                .catch((e) => {
                    this.error = e;
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
