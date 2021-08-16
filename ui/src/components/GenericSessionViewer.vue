<template>
<div>

    <b-modal id="def-modal" size="lg">
        <div>Config</div>
        <pre v-if="item.config">{{ JSON.parse(item.config) }}</pre>
        <div>Run Info</div>
        <pre v-if="item.config">{{ JSON.parse(item.runInfo) }}</pre>

    </b-modal>

    <h3><i class="fa fa-cube"></i> Generic Session</h3>

    <div class="mt-4"></div>

    <b-button-toolbar>
        <b-button variant="danger" v-if="item.status && item.status == 'RUNNING'" v-on:click="stopSession" size="sm">Abort Task</b-button>
        <b-button title="Browse Data" class="ml-2" variant="secondary" :to="`/data_browser/?path=session/${uid}`" size="sm"><i class="fa fa-folder"></i> Browse Files</b-button>

        <b-button class="ml-2" title="Show Config" variant="secondary" v-b-modal="'def-modal'" size="sm"><i class="fa fa-cog"></i> View Configuration</b-button>

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
    <div class="mt-4"></div>

    <b-container fluid>
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
                <div class="data_label">Session Type</div>
                <span>{{ item.sessionType }}</span>
            </b-col>

        </b-row>
        <div class="mt-4"></div>

        <b-row>
            <b-col>
                <div class="data_label">Environment Id</div>
                <router-link :to="`/env_spec/view/${item.envSpecId }`"> {{item.envSpecId }}</router-link>
            </b-col>

            <b-col>
                <div class="data_label">Agent Id</div>
                <span v-if="item.agent">{{item.agent.ident}}/ {{item.agent.specId}}</span>
            </b-col>

        </b-row>

        <div class="mt-4"></div>

    </b-container>

    <b-container fluid>
        <h3>Statistics</h3>
        <div class="mt-3"></div>
        <div v-if="item.summary">
            <b-row>
                <b-col>
                    {{item.summary}}
                </b-col>

            </b-row>
        </div>

    </b-container>
    <div class="mt-4"></div>
    <b-container fluid>
        <h3> Child Sessions</h3>

        <b-row>

            <b-col>
                <div>

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
                    <div>
                        No Child Sessions
                    </div>
                </div>
            </b-col>
        </b-row>
        <div class="mt-4"></div>

    </b-container>
    <div class="mt-4"></div>

</div>
</template>

<script>
// @ is an alias to /src
import axios from "axios";
import LogViewer from "./LogViewer.vue";

import {
    appConfig
} from "../app.config";
import gql from "graphql-tag";
import {
    timelength,
    timedelta,
    timepretty
} from "../funcs";

import {
    Plotly as PlotlyVue
} from 'vue-plotly'

const fields = [{
        key: "ident",
        label: "Session Id",
    },
    {
        key: "envSpecId",
        label: "Environment",
    }, {
        key: "agentId",
        label: "Agent",
    },
    {
        key: "created",
        label: "Created",
        sortable: true,
        // formatter: timepretty,
    },
    {
        key: "status",
        label: "State",
    },

    {
        key: "taskId",
        label: "Task Id",
    }
];

export default {
    name: "Session",
    components: {
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

            graphList: [],
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
                            `${appConfig.API_URL}/api/session_plots_json/${uid}/${graphItem.group}/${graphItem.value}/${graphItem.stepField}`
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
                    }
                    graphData.layout.height = 250
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
}

.plotcontainer {
    border-style: solid;
    border-width: 1px;
    border-color: #ccc;
    /* padding: 10px; */
    margin: 4px;
}
</style>
