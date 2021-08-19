<template>
<div class="page">
    <div class="page-content">
        <h1>Task Details</h1>

        <b-modal id="modal-config" title="View config" size="xl">
            <div>
                <pre>{{JSON.stringify(taskConfig,null,2)}}</pre>
            </div>
        </b-modal>

        <b-modal id="modal-summary" title="View Summary" size="xl">
            <pre v-if="task && task.summary">{{JSON.parse(task.summary)}}
            </pre>
        </b-modal>

        <b-modal id="modal-modify-source" title="TODO">
            TODO
        </b-modal>

        <b-button-toolbar size="sm" class="mr-1">
 <b-button-group size="sm" class="mr-1">
            <b-button size="sm"  :to="`/task/new/agenttask/${task.ident}`"><i class="fa fa-copy"></i> Copy</b-button>
            <b-button size="sm" :to="`/data_browser/?path=task/${uid}`"><i class="fa fa-folder"></i> Files</b-button>
            <b-button size="sm"  v-b-modal.modal-config>View Config</b-button>
            <b-button size="sm"  v-b-modal.modal-summary>View Summary</b-button>
            <b-button size="sm"  v-if="task.status && task.status == 'RUNNING'" variant="danger" v-on:click="taskControl('STOP')"><i class="fa fa-stop"></i> Abort</b-button>
            <b-button size="sm"  v-if="task.status && (task.status == 'ABORTED' || task.status == 'TERMINATED')" variant="success" v-on:click="taskControl('RUN')"><i class="fa fa-play"></i> Run</b-button>
 </b-button-group>
        </b-button-toolbar>

        <div class="mt-4"></div>
        <b-alert show variant="warning" v-if="task.status && task.statusMsg">
            <pre>{{ task.statusMsg }}</pre>
        </b-alert>

        <b-container fluid>

            <b-row>
                <b-col cols=2 class="text-center">

                    <div class="pt-2">
                        <div class="data_label">Task ID</div>
                        <span>{{ uid }}</span>
                    </div>
                    <div class="pt-2">
                        <div class="data_label">Spec ID</div>
                        <span>{{ task.specId }}</span>
                    </div>
                    <div class="pt-2">
                        <div class="data_label">Creation Time</div>
                        <span>{{ task.created }}</span>
                    </div>

                    <div class="pt-2">
                        <div class="data_label">Status</div>
                        <div v-if="task.status">
                            {{ task.status }}
                        </div>
                    </div>
                    <div class="pt-2">
                        <div class="data_label">Parent Task</div>
                        <div v-if="task.parentTask">
                            <router-link :to="`/task/view/${task.parentTaskId}`">{{task.parentTaskId}}</router-link>
                            ({{ task.parentTask.specId }})
                        </div>
                        <div v-else>No Parent</div>
                    </div>
                    <hr />
                    <div class="pt-2">
                        <h3>Primary Session</h3>
                        <div v-if="task.primarySessionId">
                            <router-link style="font-size:1.4em" :to="`/session/view/${task.primarySessionId}`">{{task.primarySessionId}}</router-link>
                        </div>
                        <div v-else>No Primary Session</div>
                    </div>

                </b-col>

                <b-col>
                    
                    <b-card title="Task Log">
                        <LogViewer :nocard="true" :logStreamUrl="`${appConfig.API_URL}/api/stream/entity_logs/task/${uid}`"> </LogViewer>
                    </b-card>
                </b-col>
            </b-row>
        </b-container>
        <b-container fluid>
            <b-row>
                <b-col>
                    <div v-if="sessionData.length>0">
                        <div class="mb-4"></div>
                        <div class="mb-4"></div>
                        <div class="mb-4"></div>
                        <b-card class="mt-3" title="Sessions">
                            <b-card-text>

                                <b-table striped hover table-busy :items="sessionData" :fields="fields" :dark="false" :small="false">
                                    <template v-slot:cell(link)="data">
                                        <!-- `data.value` is the value after formatted by the Formatter -->
                                        <router-link :to="`/session/view/${data.item.ident}`">{{ data.item.ident }}</router-link>
                                    </template>
                                </b-table>
                            </b-card-text>
                        </b-card>
                    </div>
                </b-col>
            </b-row>

            <div class="mt-4"></div>
            <b-row>
                <b-col>
                    <b-card v-if="subtaskData.length > 0" title="Sub Tasks">
                        <b-table hover table-busy :items="subtaskData" :fields="subtaskfields" :dark="false" :small="false">
                            <template v-slot:cell(link)="data">
                                <!-- `data.value` is the value after formatted by the Formatter -->
                                <router-link :to="`/task/view/${data.item.ident}`">{{
                  data.item.ident
                }}</router-link>
                            </template>
                        </b-table>
                    </b-card>
                    <div v-else>
                        No SubTasks
                    </div>
                </b-col>
            </b-row>
        </b-container>

        <div class="mt-4"></div>
    </div>
    <HelpInfo contentId="tasks"></HelpInfo>
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
    timepretty
} from "../funcs";
import gql from "graphql-tag";
import LogViewer from "../components/LogViewer.vue";

const fields = [{
        key: "link",
        label: "Session",
        sortable: true,
    },
    {
        key: "envSpecId",
        label: "Environment",
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
    }
];

const subtaskfields = [{
        key: "link",
        label: "Task",
        sortable: true,
    },

    {
        key: "specId",
        label: "Spec Id",
        sortable: true,
    },

    {
        key: "created",
        label: "Created",
        sortable: true,
    },
    {
        key: "status",
        label: "State",
    },
];

const GET_TASK = gql `
  query GetTask($ident: String!) {
    task(ident: $ident) {
      id
      ident
      status
      statusMsg
      specId

      config
      created
      summary

      primarySessionId
      parentTaskId
      parentTask {
        specId
      }
      sessions {
        pageInfo {
          startCursor
          endCursor
        }
        edges {
          node {
              id
            ident
            envSpecId
            agentId
            created
            status
            summary
          }
        }
      }
      childTasks {
        pageInfo {
          startCursor
          endCursor
        }
        edges {
          node {
              id
            ident
            created
            specId
            status
            summary
          }
        }
      }
    }
  }
`;

export default {
    name: "Task",
    components: {
        LogViewer
    },
    apollo: {
        // Simple query that will update the 'hello' vue property
        task: {
            query: GET_TASK,
            variables() {
                return {
                    ident: this.uid,
                };
            },
            pollInterval: 2000
        },
    },
    data() {
        return {
            appConfig,
            fields,
            subtaskfields,
            task: {},
            taskLogData: [],
            error: "",
            intervalTimer: null,
        };
    },
    props: {
        uid: String,
    },
    computed: {
        taskConfig() {
            if (!this.task.config) {
                return "";
            }
            return JSON.parse(this.task.config);
        },
        taskSummary() {
            if (!this.task || !this.task.summary) {
                return "{}";
            }
            return JSON.stringify(this.task.summary)
        },
        sessionData() {
            const rows = [];
            if (this.task.sessions == null || this.task.sessions.edges.length == 0) {
                return [];
            }
            for (const session of this.task.sessions.edges) {
                rows.push(session.node);
            }
            return rows;
        },

        subtaskData() {
            const rows = [];
            if (this.task.childTasks == null || this.task.childTasks.edges.length == 0) {
                return [];
            }
            for (const childTask of this.task.childTasks.edges) {
                rows.push(childTask.node);
                //
            }

            return rows;
        },
    },
    methods: {
        timepretty,
        taskControl(action) {
            if (action == "STOP") {
                axios
                    .get(`${appConfig.API_URL}/api/admin/task/stop/${this.uid}`)
                    .then((response) => {
                        // JSON responses are automatically parsed.
                        this.message = response.data["message"];
                    })
                    .catch((e) => {
                        this.error = e;
                        this.message = this.error;
                    });
            } else if (action == "RUN") {
                console.log("RUN")
                axios
                    .get(`${appConfig.API_URL}/api/admin/task/run/${this.uid}`)
                    .then((response) => {
                        // JSON responses are automatically parsed.
                        this.message = response.data["message"];
                        console.log(this.message)
                    })
                    .catch((e) => {
                        this.error = e;
                        this.message = this.error;
                        console.log(this.error)
                    })
            }
        },
    },
    // Fetches posts when the component is created.
    created() {
        console.log(this.uid);
        this.intervalTimer = setInterval(() => {
            if (!this.$apollo.queries.task.loading) {
                if (this.task.status && (this.task.status == "RUNNING" || this.task.status == "CREATED"))
                    this.$apollo.queries.task.refetch();
                else {
                    clearInterval(this.intervalTimer);
                }
            }
        }, 1000);
    },
    beforeDestroy() {
        if (this.intervalTimer) clearInterval(this.intervalTimer);
    },
};
</script>

<style>

</style>
