<template>
<div>

    <div class="mt-4"></div>

    <b-container fluid>

        <b-row>
            <b-col>
                    <div v-if="$apollo.queries.allTask.loading">Loading..</div>
                    <div v-else>
                        <div v-if="Object.keys(taskList).length > 0">
                            <b-form-checkbox-group v-model="selected">
                                <b-table hover table-busy :items="taskList" :fields="fields" :dark="false" :outlined="false" size="small">
                                    <template v-slot:cell(link)="data">
                                        <router-link :to="`/task/view/${data.item.ident}`">{{data.item.ident}}</router-link>
                                    </template>
                                    <template v-slot:cell(actions)="data">
                                        <b-button size="sm" v-if="data.item.status && data.item.status == 'RUNNING'" variant="danger" v-on:click="taskControl('STOP',data.item.ident)">Terminate</b-button>

                                    </template>
                                </b-table>
                            </b-form-checkbox-group>
                            <p>{{ error }}</p>
                        </div>

                        <div v-else>No Items Found </div>
                    </div>
            </b-col>
        </b-row>
    </b-container>
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

const fields = [{
        key: "link",
        label: "Task Id",
        sortable: true,
    },
    {
        key: "specId",
        label: "Spec Id",
        sortable: true,
        formatter: (v) => {
            if (v) return v;
            else return "session"
        }
    },
    {
        key: "created",
        label: "Creation Time",
        sortable: true,
    },
    {
        key: "status",
        label: "Status",
    },
    {
        key: "actions",
        label: "",
    },
];

const GET_ALL_TASK = gql `
  query {
    allTask(sort:CREATED_DESC) {
      pageInfo {
        startCursor
        endCursor
      }
      edges {
        node {
          ident
          specId
          created
          status
        }
      }
    }
  }
`;
export default {
    name: "Tasks",
    components: {
        // TaskList
    },
    apollo: {
        // Simple query that will update the 'hello' vue property
        allTask: GET_ALL_TASK,
    },
    data() {
        return {
            searchQuery: "",
            fields: fields,
            allTask: [],
            error: "",
            selected: [],
        };
    },

    computed: {
        taskList() {
            if (!this.allTask.edges) return [];

            return this.allTask.edges.map((task) => task.node);
        },
    },
    methods: {
        taskControl(action, uid) {
            if (action == "STOP") {
                axios
                    .get(`${appConfig.API_URL}/api/admin/task/stop/${uid}`)
                    .then((response) => {
                        // JSON responses are automatically parsed.
                        this.message = response.data["message"];
                        this.$apollo.queries.allTask.refetch()
                    })
                    .catch((e) => {
                        this.error = e;
                        this.message = this.error;
                        this.$apollo.queries.allTask.refetch()
                    });
            }
        },
    },
    // Fetches posts when the component is created.
    created() {
        //
    },
};
</script>

<style>
/* a.page-link{
   color: black;
 } */
</style>
