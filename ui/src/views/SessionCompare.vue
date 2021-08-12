<template>
<div>
    <h2><i class="fa fa-cubes"></i> Compare RL Sessions</h2>

    <b-card>
        <div v-if="Object.keys(items).length > 0">
            <b-table striped hover table-busy :items="sessionList" :fields="fields">
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
    </b-card>

    <div class="mt-4"></div>
    <b-card>

        <div>
            <hr />

            <b-container fluid>
                <b-row>
                    <b-col cols=6 v-for="graph in graphList" :key="graph.key">
                        <div v-if="graph.graphData &&
                                   graph.graphData.data &&
                                   graph.graphData.data.length > 0 &&
                                   graph.graphData.data[0] &&
                                   graph.graphData.data[0].x.length > 0">
                            <PlotlyVue :data="graph.graphData.data" :layout="graph.graphData.layout" :display-mode-bar="false"></PlotlyVue>
                        </div>
                    </b-col>
                </b-row>
            </b-container>
        </div>
    </b-card>

</div>
</template>

<script>
import axios from "axios";

import SessionList from "@/components/SessionList.vue";
import {
    appConfig
} from "../app.config";
import {
    Plotly as PlotlyVue
} from "vue-plotly";

import gql from "graphql-tag";
import {
    timedelta,
    timepretty
} from "../funcs";

const GET_SESSIONLIST = gql `
  query GetSessions($idents: String!) {
    sessionList(idents: $idents) {
      ident
      envSpecId
      created
      config
      runInfo
      status
      sessionType
      task {
        ident
      }          
      agent {
          ident
          specId
        }
      createdTimestamp
      summary 
    }
  }
`;

const fields = [{
        key: "ident",
        label: "Session Id",
    },
    {
        key: "envSpecId",
        label: "Environment",
    }, {
        key: "sessionType",
        label: "Session Type",
    },{
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
    },
    {
        key: "summary.step_count",
        label: "Total Steps",
    },
    {
        key: "summary.episode_count",
        label: "Episode Count",
    },
    {
        key: "summary.steps_per_second",
        label: "Step Per Second",
    },
];

export default {
    name: "Sessions",
    components: {
        PlotlyVue
    },
    apollo: {
        // Simple query that will update the 'hello' vue property
        sessionList: {
            query: GET_SESSIONLIST,
            variables() {
                return {
                    idents: this.uids,
                };
            },
            pollInterval: 3000
        },
    },
    data() {
        return {
            error: "",
            fields,
            sessionList: [],
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
        };
    },
    props: {
        uids: String,
    },
    methods: {
        loadGraphs() {
            this.graphList.forEach((graphItem, idx) => {
                Promise.all(this.uids.split(",").map((uid) => {
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

                    console.log(dataList.length)

                    const graph = dataList[0];

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
                    graphData.layout.height = 250
                    graphData.config = graphItem;
                    graphItem.graphData = graphData;
                    this.$set(this.graphList, idx, graphItem);

                })
            });
        }
    },

    computed: {
        items() {
            return this.sessionList;
        },
    },
    // Fetches posts when the component is created.
    created() {
        //
        this.loadGraphs()
        this.timer = setInterval(this.loadGraphs, 3000);
    },
    beforeDestroy() {
        clearInterval(this.timer);
    }
};
</script>

<style>
/* a.page-link{
   color: black;
 } */
</style>
