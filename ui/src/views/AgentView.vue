<template>
<div class="page">
    <div class="page-content">
        <b-modal id="agent-browser" size="lg" title="File Data Browser" scrollable :hide-footer="true">
            <DataBrowser :path="'/agent/' + item.ident"></DataBrowser>
            <div class="mb-5"></div>
        </b-modal>
        <h1><i class="fas fa-robot"></i> Agent: <span v-if="item">{{item.ident}}</span></h1>
        <div v-if="item && item.ident">

            <!-- <b-modal id="def-modal" size="lg">
            <pre v-if="item && item.config">{{ JSON.parse(item.config) }}</pre>
        </b-modal> -->
            <b-modal id="edit-modal" size="xl" :hide-footer="true" title="Edit Agent">
                <AgentEdit :uid="uid" @updated="configUpdated()"></AgentEdit>
            </b-modal>
            <b-modal id="meta-modal" size="lg">
                <pre v-if="item && item.meta">{{ JSON.parse(item.meta) }}</pre>
            </b-modal>

            <b-modal id="modal-publish-snapshot" title="Create Snapshot" size="lg">
                <label for="snapshot_version">Version: (Must be unique, otherwise will overwrite snapshots with same version)</label>
                <b-form-input id="snapshot_version" v-model="snapshot_version" placeholder="Enter the snapshot version." trim></b-form-input>
                <div class="mt-1"></div>

                <label for="snapshot_description">Description:</label>
                <b-form-input id="snapshot_description" v-model="snapshot_description" placeholder="Enter the snapshot description" trim></b-form-input>

                <b-button class="mt-2" variant="secondary" v-on:click="publish()" size="sm">Save</b-button>

                <hr />
                <div class="mt-2"></div>
                <b-card>
                    <h3>Current Snapshots</h3>
                    <div class="mt-2"></div>
                    <b-table :items="snapshots" :fields="pubfields">
                    </b-table>
                </b-card>
                <template v-slot:modal-footer="{ ok }">
                    <b-button variant="primary" @click="ok();">Close</b-button>
                </template>
            </b-modal>

            <b-modal id="modal-create-clone" title="Create Clone">
                <p>
                    Cloned agents will not include session history or statistics.
                </p>

                <b-button :disabled="submitting" @click="createClone()">Create Full Clone</b-button>
                <br />
                <br />
                <div class="text-center">
                    <b-button class="ml-2" v-if="clonedAgentId" :key="$route.path" :to="`/agent/view/${clonedAgentId}`" @click="clonedAgentId = null;graphList=[];$bvModal.hide('modal-create-clone')">
                        Agent Clone Id: {{clonedAgentId}}
                    </b-button>
                    <span v-else>...</span>
                </div>

            </b-modal>

            <div class="mt-4"></div>

            <!-- <b-button variant="secondary" :to="`/data_browser/?path=agent/${item.ident}`" size="sm">Browse Data</b-button> -->

            <!-- <b-button variant="secondary" v-b-modal="'def-modal'" class="ml-1" size="sm">Configuration</b-button> -->
            <b-button-toolbar>
                <b-button-group>
                    <b-button variant="primary" :to="`/task/new/agenttask/?agentUid=${uid}`" size="sm"><i class="fa fa-plus-square"></i> Assign Task</b-button>

                    <b-button variant="secondary" v-b-modal="'edit-modal'" size="sm"><i class="fa fa-edit"></i> Configure</b-button>
                    <b-button title="Create Snapshot" variant="secondary" v-b-modal="'modal-publish-snapshot'" @click="loadSnapshotList()" size="sm"><i class="fa fa-camera-retro"></i> Snapshot</b-button>
                    <b-button title="Create Clone" variant="secondary" v-b-modal="'modal-create-clone'" size="sm"><i class="fa fa-clone"></i> Clone</b-button>
                    <b-button variant="warning" v-if="item.job_data && item.job_data.state == 'RUNNING'" v-on:click="agentControl('SHUTDOWN')" size="sm">Shutdown</b-button>
                    <b-button variant="danger" v-if="item.job_data && item.job_data.state == 'RUNNING'" v-on:click="agentControl('KILL')" size="sm">Kill</b-button>
                    <b-button variant="secondary" v-b-modal.agent-browser size="sm"><i class="fa fa-file"></i> Files</b-button>
                    <b-button variant="secondary" v-b-modal="'meta-modal'" size="sm"><i class="fa fa-info-circle"></i> Metadata</b-button>
                    <b-button variant="info" title="Move to archive" v-if="item && !item.archived"  @click="updateArchive(true)" size="sm"><i class="fa fa-trash"></i> Move to Archive</b-button>
                    <b-button title="restore from archive" v-if="item && item.archived" variant="secondary" @click="updateArchive(false)" size="sm"><i class="fa fa-trash-restore"></i> Restore from Archive</b-button>
                </b-button-group>
            </b-button-toolbar>

            <div class="mt-4"></div>
            <b-modal id="errorbox" size="xl">
                <b-alert show v-if="error != null" variant="danger">
                    <pre>{{error}}</pre>
                    <pre>{{traceback}}</pre>
                </b-alert>
            </b-modal>
            <AgentCard v-if="item" :agent="item" @update="refetch()"></AgentCard>
            <hr />
            <div class="mt-4"></div>

            <h3>Session History</h3>
            <div class="mt-4 mb-4">
                Total Steps Experienced: {{lifeSteps}}
            </div>
            <div style="display: block; position: relative;height:280px;overflow: auto;">
                <b-button :disabled="selected.length <= 1" v-on:click="runCompare" variant="info" size="sm">
                    <span>Compare: {{ selected.length }}</span>

                </b-button>
                <b-form-checkbox-group v-model="selected">

                    <b-table show-empty empty-text="No Sessions Found" hover table-busy :items="rowData" :fields="fields" :dark="false" :small="false" :borderless="false" sortBy="created" :sortDesc="true">
                        <template v-slot:cell(selector)="data">

                            <b-form-checkbox :value="data.item.ident"></b-form-checkbox>
                        </template>
                        <template v-slot:cell(parentlink)="data">
                            <!-- `data.value` is the value after formatted by the Formatter -->
                            <router-link :to="`/session/view/${data.item.parentSessionId}`">{{data.item.parentSessionId}}</router-link>
                        </template>
                        <template v-slot:cell(link)="data">
                            <!-- `data.value` is the value after formatted by the Formatter -->
                            <router-link :to="`/session/view/${data.item.ident}`">{{data.item.ident}}</router-link>
                        </template>
                        <template v-slot:cell(status)="data">
                            <!-- `data.value` is the value after formatted by the Formatter -->
                            <span>{{data.item.status}}</span>
                            <b-button class="ml-1" variant="danger" v-if="data.item.status && data.item.status == 'RUNNING'" v-on:click="stopSession(data.item.task.ident)" size="sm">Abort</b-button>

                        </template>
                        <template v-slot:cell(steps)="data">
                            <!-- `data.value` is the value after formatted by the Formatter -->
                            <span v-if="data.item.summary.step_count">{{data.item.summary.step_count.toLocaleString('en-US',
                            {
                            useGrouping: true
                            })}}</span>
                        </template>
                        <template v-slot:cell(episodes)="data">
                            <!-- `data.value` is the value after formatted by the Formatter -->
                            <span v-if="data.item.summary.episode_count">{{data.item.summary.episode_count.toLocaleString('en-US',
                            {
                            useGrouping: true
                            })}}</span>

                        </template>
                        <template v-slot:cell(runtime)="data">
                            <!-- `data.value` is the value after formatted by the Formatter -->
                            <span v-if="data.item.summary.runtime">{{data.item.summary.runtime.toLocaleString('en-US',
                            {
                            useGrouping: true
                            })}}</span>
                        </template>
                    </b-table>
                </b-form-checkbox-group>

            </div>
            <div class="mt-4"></div>

            <hr />
            <h3>Statistics</h3>

            <hr />
            <b-container fluid>
                <b-row>
                    <b-col>
                        <b-form-group label="History" v-slot="{ ariaDescribedby }">
                            <b-form-radio-group v-model="plotStepMax" @click="loadGraphs()" :options="plotStepMaxOptions" :aria-describedby="ariaDescribedby" button-variant="secondary" size="sm" name="radio-btn-outline" buttons></b-form-radio-group>
                            <span> Showing steps: {{plot_total_count-plot_actual_count}} to {{plot_total_count}}</span>
                        </b-form-group>
                    </b-col>
                    <b-col>
                        <b-form-group label="Bin Size" v-slot="{ ariaDescribedby }" class="pull-right">
                            <b-form-radio-group v-model="plotBinSize" @click="loadGraphs()" :options="plotBinSizeOptions" :aria-describedby="ariaDescribedby" button-variant="secondary" size="sm" name="radio-btn-outline" buttons></b-form-radio-group>
                        </b-form-group>
                    </b-col>
                </b-row>
                <div class="mt-4"></div>
                <b-row>
                    <b-col cols=3 v-for="graph in graphList" :key="graph.key">

                        <div v-if="graph.data && graph.layout && (graph.data.length > 0 ) && graph.data[0] && graph.data[0].x.length > 0">
                            <PlotlyVue bgcolor="#000" :data="graph.data" :layout="graph.layout" :display-mode-bar="false"></PlotlyVue>
                        </div>

                    </b-col>
                    <b-col v-if="graphList.length==0">
                        No log data found
                    </b-col>
                </b-row>
            </b-container>
        </div>
    </div>
    <HelpInfo contentId="agents"></HelpInfo>
</div>
</template>

<script>
// @ is an alias to /src
import axios from "axios";
import {
    appConfig
} from "../app.config";
import {
    timedelta,
    timepretty,
    timelength,
    formatNum
} from "../funcs";
import gql from "graphql-tag";
import {
    Plotly as PlotlyVue
} from 'vue-plotly'

const fields = [{
        key: "selector",
        label: "",
        sortable: false,
    }, {
        key: "link",
        label: "Session",
        sortable: false,
    },
    {
        key: "parentlink",
        label: "Parent",
        sortable: true,
    },
    {
        key: "envSpecId",
        label: "Environment",
    },
    {
        key: "sessionType",
        label: "Session Type",
    },
    {
        key: "status",
        label: "Status",
        sortable: true,
        // formatter: timepretty,
    },
    {
        key: "runtime",
        label: "Run Time (Sec)",
        sortable: true
    },

    // {
    //     key: "statusTimestamp",
    //     label: "Last Update",
    //     sortable: true,
    //      formatter: (d=>d.toLocaleString('en-US',{ weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })),
    // },

    {
        key: "steps",
        label: "Total Steps",
        sortable: true,
        // formatter: timepretty,
    },
    {
        key: "episodes",
        label: "Completed Episodes",
        sortable: true,
        // formatter: timepretty,
    }
];

const pubfields = [

    {
        key: "snapshot_version",
        label: "Version",
    },

    {
        key: "creation_time",
        label: "Created",

    },

    {
        key: "submitter_id",
        label: "Submitter Id",

    },

    {
        key: "snapshot_description",
        label: "Description",

    }

];
const GET_AGENT = gql `
  query GetAgent($ident: String!) {
    item: agent(ident: $ident) {
      id
      ident
      seed
      specId
      spec{
          id
          ident
          displayedName
      }
      config
      meta
      created
      notes
      archived
      tags{
          edges{
              node{
                  id
                  tagId
              }
          }
      }
      lastCheckpoint
      components{
        edges{
          node{
            ident
            name
            specId
            spec{
              ident
              category
            }
          }
        }
      }
      sessions(first: 100) {
        pageInfo {
          startCursor
          endCursor
        }
        edges {
          node {
            ident
            envSpecId
            task {
              ident
            }
            sessionType
            created
            status
            statusTimestamp
            archived
            summary 
            parentSessionId
          }
        }
      }
    }
  }
`;
import AgentCard from "../components/AgentCard.vue";
import DataBrowser from "./DataBrowser.vue";

import AgentEdit from "../components/AgentEdit.vue";

export default {
    name: "Agent",
    components: {
        AgentCard,
        AgentEdit,
        PlotlyVue,
        DataBrowser
    },
    apollo: {
        // Simple query that will update the 'hello' vue property
        item: {
            query: GET_AGENT,
            variables() {
                return {
                    ident: this.uid,
                };
            },
            pollInterval: 4000
        },
    },
    data() {
        return {
            clonedAgentId: null,
            cloneError: null,
            submitting: false,

            taskDetailsList: [],
            fields,
            pubfields,
            snapshot_description: "",
            snapshot_version: "0-dev",
            snapshots: [],
            plot_actual_count: 0,
            plot_total_count: 0,
            plotBinSize: 0,
            plotStepMax: 0,
            selected: [],
            traceback: null,

            componentFields: [{
                    key: "name",
                    label: "Name",
                }, {
                    key: "specId",
                    label: "Spec Id",
                },
                {
                    key: "spec.category",
                    label: "Type",
                }
            ],
            graphList: [],
            error: null,
            item: {},
            timer: null,
            timelength,
            formatNum
        };
    },
    props: {
        uid: String,
    },
    computed: {
        lifeSteps() {
            if (this.item.sessions == null) {
                return 0;
            }
            let totalSteps = 0
            for (const session of this.item.sessions.edges) {
                if (session.node.archived != true) {
                    if (session.node.parentSessionId == null)
                        totalSteps += session.node.summary.step_count
                }
            }
            return totalSteps

        },
        config() {
            return JSON.parse(this.item.config)
        },
        meta() {
            return JSON.parse(this.item.meta)
        },

        plotStepMaxOptions() {
            if (this.plot_total_count == 0) {
                return []
            }
            const places = this.plot_total_count.toString().length - 1

            const options = []
            options.push({
                text: `All (${this.plot_total_count})`,
                value: '0'
            })
            for (let p = places; p > 2; p--) {
                const val = Math.pow(10, p)
                options.push({
                    'text': `last ${val}`,
                    'value': `${val}`
                })

            }
            return options

        },
        plotBinSizeOptions() {
            if (this.plot_total_count == 0) {
                return []
            }
            const places = this.plot_total_count.toString().length - 2

            const options = []
            options.push({
                text: `Auto`,
                value: '0'
            })
            if (places > 1) {
                for (let p = places; p > 0; p--) {
                    const val = Math.pow(10, p)
                    options.push({
                        'text': `${val}`,
                        'value': `${val}`
                    })

                }
            }
            options.push({
                text: `5`,
                value: '5'
            })
            options.push({
                text: `1`,
                value: '1'
            })
            return options

        },
        rowData() {
            const rows = [];
            if (this.item.sessions == null) {
                return rows;
            }

            for (const session of this.item.sessions.edges) {
                if (session.node.archived != true) {
                    rows.push(session.node);
                }
            }

            return rows;
        },

    },
    methods: {
        updateArchive(archive) {
            // We save the user input in case of an error
            // const newTag = this.newTag
            // // We clear it early to give the UI a snappy feel
            // this.newTag = ''
            // Call to the graphql mutation
            this.$apollo.mutate({
                // Query
                mutation: gql `mutation archiveMutation($id:String!,$archive:Boolean!) 
                {
                    agentSetArchive(id:$id, archive:$archive){
                        success
                        }
                }`,
                // Parameters
                variables: {
                    id: this.item.id,
                    archive: archive
                },

            }).then((data) => {
                this.refetch()
            }).catch((error) => {
                // Error
                console.error(error)
                // We restore the initial user input
            })
        },
        timepretty,

        refetch() {
            this.$apollo.queries.item.refetch()

        },
        runCompare() {
            this.$router.push({
                path: `/session/compare?uids=` + this.selected.join(","),
            });
        },
        configUpdated() {
            console.log("ConfigUPdate")
            this.refetch();
            this.$bvModal.hide("edit-modal");

            //TODO: close modal

        },
        // saveConfig(config){

        //     console.log(JSON.stringify(config))
        //     this.$apollo.mutate({
        //         // Query
        //         mutation: gql `mutation agentConfigMutation($id:String!,$config:String!) 
        //         {
        //             agentSetConfig(id:$id, config:$config){
        //                 success
        //                 }
        //         }`,
        //         // Parameters
        //         variables: {
        //             id: this.agent.id,
        //             config: JSON.stringify(config)
        //         },

        //     }).then((data) => {
        //         this.refetch()
        //     }).catch((error) => {
        //         // Error
        //         console.error(error)
        //         // We restore the initial user input
        //     })
        // },
        stopSession(taskId) {
            axios
                .get(
                    `${appConfig.API_URL}/api/admin/task/stop/${taskId}`
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
        },

        agentControl(action) {
            axios
                .get(`${appConfig.API_URL}/api/agent_control/${action}?uid=${this.uid}`)
                .then((response) => {
                    // JSON responses are automatically parsed.
                    this.message = response.data["message"];
                })
                .catch((e) => {
                    this.error = e;
                    this.message = this.error;
                    this.$bvModal.show("errorbox")
                });
        },
        loadSnapshotList() {
            if (this.item && this.item.seed) {
                axios
                    .get(`${appConfig.API_URL}/api/snapshots/agent/list/${this.item.seed}`)
                    .then((response) => {
                        this.snapshots = response.data["items"]
                    })
                    .catch((e) => {
                        this.error = e;
                        console.log(e)
                    });
            }
        },
        loadGraphs() {
            axios
                .get(
                    `${appConfig.API_URL}/api/agent_plots_json/${this.uid}?max_steps=${this.plotStepMax}&bin_size=${this.plotBinSize}`
                )
                .then((response) => {
                    const graphdata = response.data.plot_data;
                    if (graphdata == null) {
                        this.error = response.data.error
                        this.traceback = response.data.traceback
                        console.log(this.error)
                        console.log(this.traceback)
                        return
                    }

                    this.plot_actual_count = response.data.actual_count;
                    this.plot_total_count = response.data.total_count;

                    this.graphList = []

                    Object.keys(graphdata).forEach((statName) => {
                        const graphData = {}
                        const graph = graphdata[statName]
                        if (graph.data) {
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
                            graphData.layout.font = {
                                color: "rgba(200,200,200,1)"
                            }
                            graphData.layout.plot_bgcolor = "rgba(0,0,0,0.1)";
                            graphData.layout.paper_bgcolor = "rgba(0,0,0,0.1)";
                            graphData.layout.yaxis = {

                                "gridcolor": "rgba(200,200,200,0.25)",
                                "gridwidth": 1,

                            }
                            graphData.layout.xaxis = {

                                "gridcolor": "rgba(200,200,200,0.25)",
                                "gridwidth": 1,

                            }
                            graphData.layout.height = 250
                            this.graphList.push(graphData)
                        }
                    })

                })
                .catch((e) => {
                    console.log(e);
                    this.error = e;
                    this.$bvModal.show("errorbox")
                });
        },

        publish() {
            const outgoingData = {
                snapshot_version: this.snapshot_version,
                snapshot_description: this.snapshot_description,
                agent_id: this.uid

            }
            console.log(JSON.stringify(outgoingData, null, 2))
            this.error = null
            this.traceback = null
            axios
                .post(`${appConfig.API_URL}/api/snapshot/publish`, outgoingData)
                .then((response) => {
                    const data = response.data["item"];
                    if ("snapshot_data" in data) {
                        console.log("Snapshot Result: " + JSON.stringify(data['snapshot_data']));
                    } else {
                        console.log("ERROR in response " + JSON.stringify(data));
                        this.error = data['error'];
                        this.$bvModal.show("errorbox")
                    }
                    this.traceback = data["traceback"];

                    this.submitting = false;
                    this.loadSnapshotList()
                })
                .catch(function (error) {
                    this.error = error;
                    this.submitting = false;
                });
        },
        createClone() {
            this.cloneError = null
            this.submitting = true
            axios
                .get(`${appConfig.API_URL}/api/agent/clone/${this.uid}`)
                .then((response) => {
                    console.log(JSON.stringify(response.data, null, 2))
                    if ("uid" in response.data) {
                        this.clonedAgentId = response.data.uid

                        // this.$router.push({
                        //         path: '/agent/view/' + ,
                        //         key:this.$route.path
                        //     })
                    } else {
                        this.cloneError = response.message
                    }
                    this.submitting = false;
                })
                .catch(function (error) {
                    this.cloneError = error;
                    this.submitting = false;
                });
        },
    },
    // Fetches posts when the component is created.
    created() {
        console.log(this.uid);
        this.loadSnapshotList()
        this.timer = setInterval(this.loadGraphs, 2000);
        this.loadGraphs()
    },
    beforeDestroy() {

        clearInterval(this.timer);
    },

};
</script>

<style>
/* a.page-link{
   color: black;
 } */
</style>
