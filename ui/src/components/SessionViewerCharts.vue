<template>
<div>
    <h1><i class="fa fa-cube"></i> Session</h1>
    <hr />

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
        <h3>Task Log</h3>
        <LogViewer :logStreamUrl="`${appConfig.API_URL}/api/stream/entity_logs/task/${task.ident}`"> </LogViewer>
    </b-collapse>

    <b-collapse id="sessionlogs" class="mt-2">
        <h3>Session Log</h3>
        <LogViewer :logStreamUrl="`${appConfig.API_URL}/api/stream/entity_logs/session/${uid}`"> </LogViewer>
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

                    <div>
                        <div>Session Id</div>
                        <h3>{{item.ident}} </h3>
                    </div>
                </b-col>
                <b-col>

                    <div class="data_label">Parent Session Id</div>
                    <div v-if="parentSessionId">
                        <router-link :to="`/session/view/${parentSessionId}`">{{parentSessionId}}</router-link>

                    </div>
                    <div v-else>
                        No Parent Session
                    </div>

                </b-col>
                <b-col>

                    <div>
                        <div class="data_label">Task Id</div>
                        <span>
                            <router-link :to="`/task/view/${task.ident}`">{{
                    task.ident
                  }}</router-link>
                        </span>
                    </div>
                </b-col>
                <b-col>

                    <div>
                        <div class="data_label">State</div>
                        <span v-if="item.status=='RUNNING'" style="color:green;font-weight:600">{{ item.status }}</span>
                        <span v-else>{{ item.status }}</span>
                    </div>
                </b-col>
                <b-col>

                    <div>

                        <div class="data_label ">Creation Time</div>
                        <span>{{ item.created }}</span>
                    </div>
                </b-col>
                <b-col>

                    <div>
                        <div class="data_label">Runtime (seconds)</div>
                        <span v-if="item && item.summary">{{timelength(formatNum(item.summary.runtime,4))}}</span>

                    </div>
                </b-col>

            </b-row>
        </b-card>
        <b-row class='text-center'>

            <b-col class="d-flex justify-content-around">
                <b-card>

                    <h3>Environment</h3>
                    <div class="text-center" style="height:320px;">
                        <div>
                            <router-link :to="`/env_spec/view/${item.envSpecId }`"> {{item.envSpecId }}</router-link>
                        </div>
                        <div class="mt-2">

                            <div v-if="!playingPreview">
                                <img v-if="item.envSpec && item.envSpec.environment && item.envSpec.environment.ident" :src="`${appConfig.API_URL}/api/env_preview_image/${item.envSpec.environment.ident}`" alt="" style="width:auto;height:250px;" />
                            </div>

                            <div v-else>
                                <div v-if="item.status && item.status == 'RUNNING'">
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

                            <div class="">
                                <div v-if="maxEpisode">
                                    <router-link :to="`/episode/view/${item.ident}?episodeId=${maxEpisode}`">Total Recorded episodes: {{ totalRecordedEpisodes }}</router-link>
                                </div>
                                <div v-else>
                                    No Episodes Recorded
                                </div>
                            </div>
                        </div>
                    </div>
                </b-card>
            </b-col>

            <b-col class='d-flex justify-content-around'>
                <b-card>
                    <h3>Agent</h3>
                    <AgentCardSmall :agent="item.agent"></AgentCardSmall>

                </b-card>
            </b-col>

        </b-row>
    </b-container>

    <div class="mt-3"></div>
    <b-card title="Statistics">

        <b-container fluid>

            <div v-if="item.summary">
                <b-row>
                    <b-col>

                        <div>
                            <div class="stat_label">Avg Reward/Episode</div>
                            <span class="stat_value">{{
                          numberToString(formatNum(item.summary.mean_reward_per_episode, 4))
                        }}</span>
                        </div>
                    </b-col>
                    <b-col>
                        <div>
                            <div class="stat_label">Avg Reward/Step</div>
                            <span class="stat_value">{{
                          numberToString(formatNum(item.summary.mean_reward_per_step, 4))
                        }}</span>
                        </div>
                    </b-col>
                    <b-col>
                        <div>
                            <div class="stat_label">
                                Avg Reward/Step (Windowed)
                            </div>
                            <span class="stat_value">{{
                          numberToString(formatNum(item.summary.reward_mean_windowed, 4))
                        }}</span>
                        </div>
                    </b-col>
                    <b-col>
                        <div>
                            <div class="stat_label">Steps/Episode</div>
                            <span class="stat_value">{{
                          numberToString(formatNum(item.summary.mean_steps_per_episode, 4))
                        }}</span>
                        </div>
                    </b-col>

                    <b-col>

                        <div>
                            <div class="stat_label">Total Episodes</div>
                            <span class="stat_value">{{
                          numberToString(item.summary.episode_count)
                        }}</span>
                        </div>
                    </b-col>
                </b-row>
                <div class="mt-2"></div>
                <b-row>

                    <b-col>
                        <div>
                            <div class="stat_label">Total Reward</div>
                            <span class="stat_value">{{
                          numberToString(formatNum(item.summary.reward_total,8))
                        }}</span>
                        </div>
                    </b-col>

                    <b-col>
                        <div>
                            <div class="stat_label">Total Steps</div>
                            <span class="stat_value">{{
                          numberToString(item.summary.step_count)
                        }}</span>
                        </div>

                    </b-col>
                    <b-col>

                        <div>
                            <div class="stat_label">Steps/Second</div>
                            <span class="stat_value">{{
                          numberToString(formatNum(item.summary.steps_per_second, 4))
                        }}</span>
                        </div>

                    </b-col>
                    <b-col>

                        <div>
                            <div class="stat_label">Best Episode Reward Total</div>
                            <span class="stat_value">{{numberToString(formatNum(item.summary.best_ep_reward_total, 4))}}</span>
                        </div>

                    </b-col>

                    <b-col>

                        <div>
                            <div class="stat_label">Best Episode Reward Mean over Window</div>
                            <span class="stat_value">{{numberToString(formatNum(item.summary.best_ep_reward_mean_windowed, 4))}}</span>
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
        </b-container>
        <div class="mt-4"></div>
        <hr />
        <div class="mt-4"></div>
        <b-container fluid>
            <b-row>

                <b-col cols=3 v-for="(graph,i) in graphListResults" :key="i">
                    <div v-if="graph && graph.graphData">

                        <div v-if="graph.graphData.data && graph.graphData.layout && (graph.graphData.data.length > 0 ) && graph.graphData.data[0] && graph.graphData.data[0].x.length > 0">
                            <!-- <PlotlyVue :data="graph.graphData.data" :layout="graph.graphData.layout" :display-mode-bar="false"></PlotlyVue> -->
                            <LineChart :chart-data="graph.graphData.chartData"></LineChart>

                        </div>

                    </div>

                </b-col>
            </b-row>

        </b-container>
    </b-card>
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
            maxEpisode: "",
            activeGraphs: {},
            appConfig,
            playingPreview: false,
            imageURL: "placeholder.jpg",
            videoURL: "",
            graphListResults: [],
            graphList: [{
                    key: "episode_reward_total",
                    group: "ep_stats",
                    value: "episode_reward_total",
                    stepField: "episode_count",
                    color: 'rgb(0, 160, 0)',
                    title: "Total Reward per Episode",
                    stats: [{}]
                },

                {
                    key: "episode_step_count",
                    group: "ep_stats",
                    value: "episode_step_count",
                    stepField: "episode_count",
                    color: 'rgb(255, 160, 0)',

                    stats: [{}]
                },

                {
                    key: "reward_total",
                    group: "step_stats",
                    value: "reward_total",
                    stepField: "step_count",
                    color: 'rgb(255, 100, 100)',
                    stats: [{}]
                },
                {
                    key: "step_latency_mean_windowed",
                    group: "step_stats",
                    value: "step_latency_mean_windowed",
                    stepField: "step_count",
                    color: 'rgb(55, 128, 191)',
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
        numberToString,
        formatNum,
        getImageId(uid) {
            if (uid) {

                let id = parseInt(uid.split("-")[1]);
                return id % 19;
            } else ""
        },

        imageError(event) {
            console.log(event);
            this.imageURL = "placeholder.jpg";
        },
        refreshData() {
            this.playingPreview = (this.item.status == "RUNNING")
            if (this.item.status == null || (this.item.status && this.item.status == "RUNNING")) {
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
            // this.graphListResults = []

            this.graphList.forEach((graphItem, idx) => {
                let url = `${appConfig.API_URL}/api/session_plots_json/${this.uid}/${graphItem.group}/${graphItem.value}/${graphItem.stepField}`;
                // console.log("Fetching: "+ url);

                axios
                    .get(
                        url
                    )
                    .then((response) => {
                        const graph = response.data;
                        const graphData = {}

                        if (graph == undefined) {
                            return
                        }

                        if (graph.data)
                            graph.data.line = {
                                color: graphItem.color
                            }

                        graphData.data = [graph.data];

                        if (!graph.layout) return
                        graphData.layout = graph.layout;
                        graphData.layout.autosize = true
                        graphData.layout.margin = {
                            t: 25,
                            l: 25,
                            r: 25,
                            b: 25
                        }
                        // console.log(JSON.stringify(graph.data.x,null,2))
                        graphData.layout.height = 250
                        graphData.config = graphItem
                        graphItem.graphData = graphData
                        var points = graph.data.x.map(function (e, i) {
                            return {
                                x: e,
                                y: graph.data.y[i]
                            };
                        });
                        console.log(JSON.stringify(points,null,2))
                        graphItem.graphData.chartData = {

                            datasets: [{
                                fill:false,
                                label: 'Data One',
                                backgroundColor: '#f87979',
                                data: points
                            }]
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
}

.plotcontainer {
    border-style: solid;
    border-width: 1px;
    border-color: #ccc;
    /* padding: 10px; */
    margin: 4px;
}
</style>
