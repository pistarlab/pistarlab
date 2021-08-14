<template>
<div>

    <b-container fluid>
        <b-row>
            <b-col cols="8">
                <h3>
                    Agent Session Task
                </h3>

                <div class="mt-2">
                    Launch one or more Agents on a single Environment.
                </div>
            </b-col>
            <b-col cols="4" align-self="center">
                <div class="text-right">
                    <b-button-group class="mr-1">
                        <b-button size="sm" :to="`/task/new/agenttask`" @click="click()" variant="primary">Select</b-button>
                    </b-button-group>
                </div>

            </b-col>
        </b-row>
    <div class="mt-4"></div>
<hr/>
        <b-row>
            <b-col cols="8">
                <h3>
                    Hyper Parameter Experiment
                </h3>
                <strong>WARNING: This feature has limited functiunality. UI is currently Under development</strong>
                <div class="mt-2">
                    
                    Launch a hyper parameter experiments for a specified Agent Spec and Environment.
                    
                </div>
            </b-col>
            <b-col cols="4" align-self="center">
                <div class="text-right">
                    <b-button-group class="mr-1">
                        <b-button size="sm" :to="`/task/new/hyperparam_exp`" variant="primary">Select</b-button>
                    </b-button-group>
                </div>
            </b-col>
        </b-row>
    </b-container>
    <div class="mt-4"></div>
    <hr/>
    <b-container>

        <b-link size="sm" v-b-toggle.manualtask variant="secondary">Manual Task Launchers (Advanced)</b-link>
        <b-collapse id="manualtask">
            <div v-if="$apollo.queries.taskSpecs.loading">Loading..</div>
            <div v-if="taskSpecs && taskSpecs.length > 0">
                <p class="m-2">
                    Launch a task by manually creating a configuration.
                </p>
                <div v-for="item in taskSpecs" :key="item.id" class="pt-4">
                    <b-row>

                        <b-col cols="8">
                            <h3>
                                {{item.displayedName}}
                            </h3>
                            <div class="">
                                <b-alert size="sm" v-if="item.disabled" show variant="warning">disabled</b-alert>
                            </div>
                            <!-- <div class=" data_label">Class/Module</div>
                      <span>{{ item.classHame }}/{{ item.module }}</span>
-->
                            <div>
                                <span class="data_label mt-1">Extension: </span> {{ item.extensionId }}/{{ item.version }}
                            </div>
                            <div class="mt-2">
                                {{ item.description }}
                            </div>
                        </b-col>
                        <b-col cols="4" align-self="center">
                            <div class="text-right">
                                <b-button-group class="mr-1">
                                    <b-button size="sm" :disabled=item.disabled :to="`/task/new/${item.ident}`" variant="primary">Select</b-button>
                                </b-button-group>
                            </div>

                        </b-col>

                    </b-row>

                    <hr />
                </div>
            </div>
            <div v-else>
                <b-row>
                    <b-col>{{ message }}</b-col>
                </b-row>
            </div>

        </b-collapse>

    </b-container>
</div>
</template>

<script>
// @ is an alias to /src
import axios from "axios";
import {
    appConfig
} from "../app.config";
import gql from "graphql-tag";

export default {
    name: "Tasks",
    components: {
        // SessionList
    },
    apollo: {
        // Simple query that will update the 'hello' vue property
        taskSpecs: gql `
      query {
        taskSpecs: taskSpecs {
          id
          ident
          displayedName
          disabled
          extensionId
          description
          version
        }
      }
    `,
    },
    data() {
        return {
            taskSpecs: [],
            searchQuery: "",
            error: "",
            message: "Loading...",
        };
    },

    computed: {
        // items() {
        //   return this.qlitems
        // }
    },
    // Fetches posts when the component is created.
    created() {
        //
    },
};
</script>
