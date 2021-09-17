<template>
<div class="page">
    <div class="page-content">
        <h1><i class="fa fa-gamepad"></i> Environments</h1>
        <div class="mt-4"></div>

        <b-modal id="modal-addcustom">
            TODO
        </b-modal>

        <div class="mt-4"></div>
        <b-modal id="modal-selected" footerClass="p-2 border-top-0" :title="`${selectedEnvironment.displayedName}`" size="lg" scrollable hide-footer>

            <b-container fluid>

                <b-row>
                    <b-col>

                        <div>
                            <span class="data_label  mt-2">Categories: </span>
                            <span class="data_label">{{ selectedEnvironment.categories }}</span>
                        </div>
                        <div>
                            <span class="data_label  mt-2">Version: </span>
                            <span class="data_label">{{ selectedEnvironment.version }}</span>
                        </div>

                        <div>
                            <span class="data_label  mt-2">Extension: </span>
                            <span class="data_label">{{ selectedEnvironment.extensionId }}: {{ selectedEnvironment.extensionVersion }}</span>
                        </div>

                        <div>
                            <span class="data_label mt-2">Description: </span>
                            <span>{{selectedEnvironment.description}}</span>
                        </div>
                    </b-col>
                    <b-col>
                        <img :src="`${appConfig.API_URL}/api/env_preview_image/env_${selectedEnvironment.ident}`" alt="" style="max-height:200px;" />
                    </b-col>
                </b-row>
            </b-container>

            <div class="mt-4">
            </div>
            <h4>Environment Specs</h4>

            <b-container fluid>
                <div v-for="spec in selectedEnvironment.specs" :key="spec.ident">
                    <b-row>
                        <b-col cols=3 class="text-center  align-middle">
                            <div class="mt-2 mb-2">
                                <router-link :to="`/env_spec/view/${spec.ident}`">
                                    <b-card-img :src="`${appConfig.API_URL}/api/env_preview_image/${spec.ident}`" alt="No Image Found" style="width:100px;"></b-card-img>
                                </router-link>
                            </div>

                        </b-col>
                        <b-col>
                            <div>
                                <span>
                                    <router-link :to="`/env_spec/view/${spec.ident}`"> {{ spec.displayedName }}
                                    </router-link>
                                </span>
                            </div>
                            <div>
                                <span class="data_label mt-1">SpecId: </span>
                                <span>
                                    {{ spec.ident }}

                                </span>
                            </div>
                            <div>
                                <span class="data_label mt-1">Type: </span>
                                <span>{{spec.envType}}</span>
                            </div>

                            <div>
                                <span class="data_label mt-1">Tags: </span>
                                <span>{{spec.tags}}</span>
                            </div>
                            <div>
                                <span class="data_label mt-1">Description: </span>
                                <span>{{spec.description}}</span>
                            </div>
                        </b-col>

                        <b-col cols=2 class="my-auto">

                            <div>
                                <b-button variant="primary"  title="Assign to agent" :to="`/task/new/agenttask/?envSpecId=${spec.ident}`" size="sm">Assign</b-button>

                            </div>
                            <!-- <div v-if="selectedEnvironment.ident == 'landia'">
                                <b-button title="Play in user interactive mode" variant="primary" @click="launchHumanMode(spec.ident)" size="sm"><i class="fa fa-user-circle"></i> Play</b-button>

                            </div> -->
                        </b-col>
                    </b-row>
                    <div class="mt-1"></div>

                    <hr />
                </div>
            </b-container>
        </b-modal>
        <div class="mt-4"></div>
        <b-container fluid>
            <div v-if="$apollo.queries.environments.loading">Loading..</div>

            <b-row>
                <b-col class="my-auto">
                    <span class='ml-5 h6'> Collections: </span>
                    <b-form-radio-group class='ml-2' size="sm" v-model="selectedCollection" :options="collections" buttons></b-form-radio-group>
       
                    <b-form-input class='ml-auto' v-model="searchtext" placeholder="Search Environments" style="width:250px;"></b-form-input>
                </b-col>
            </b-row>
            <div>
                <div v-if="items.length > 0">
                    <div class="mt-4"></div>
                    <b-row>
                        <b-col class="d-flex flex-wrap justify-content-center  mb-4">
                            <span v-for="(item, idx) in items" v-bind:key="idx" class="m-2" @click="selectGroup(idx)" v-b-modal.modal-selected style="min-width:150px">
                                <EnvironmentCard :item="item"></EnvironmentCard>

                            </span>
                        </b-col>
                    </b-row>
                </div>
                <div v-else class="m-5 text-center">
                    {{message}}
                </div>
            </div>

        </b-container>
        <br />
        <div class="mt-4"></div>
    </div>
    <HelpInfo contentId="envs"></HelpInfo>
</div>
</template>

<script>
// @ is an alias to /src
import gql from "graphql-tag";
import {
    appConfig
} from "../app.config";
import EnvironmentCard from "../components/EnvCardGroup.vue";
import axios from "axios";
import {
    GET_ALL_ENVS
} from "../queries"

export default {
    name: "Env",
    components: {
        EnvironmentCard
    },
    apollo: {
        environments: GET_ALL_ENVS,
    },
    data() {
        return {
            environments: [],
            searchQuery: "",
            selectedEnvironment: {},
            message: "No Environments found.",
            appConfig,
            searchtext: "",
            selectedCollection: null
        };
    },
    methods: {
        selectGroup(idx) {
            this.selectedEnvironment = this.items[idx];

        },
        createCollections(envs) {
            var colls = {}
            envs.forEach(element => {
                var els = []
                if (element.collection in colls) {
                    els = colls[element.collection]
                }
                els.push(element)
                colls[element.collection] = els

            });

            return colls

        },
        launchHumanMode(specId){
          this.loading = true

            axios
                .get(`${appConfig.API_URL}/api/env/human_mode/${specId}`)
                .then((response) => {
                    console.log("Run success " + response.data)
                    
                    this.loading = false

                })
                .catch((e) => {
                    console.log("Run error")
                    this.error = e;
                    this.loading = false
                });

        }
    },

    computed: {
        allitems() {
            if (this.environments.length == 0) return [];
            else {
                return this.environments.edges.map((v) => v.node).filter((v) => !v.disabled);
            }
        },
        items() {
            let envs = this.allitems;
            if (this.selectedCollection != null) {
                envs = envs.filter(v =>
                    v.collection != null && v.collection == this.selectedCollection
                )
            }

            if (this.searchtext != "") {
                return envs.filter((v) =>
                    v.displayedName.toLowerCase().includes(this.searchtext.toLowerCase())
                )
            } else {
                return envs
            }

        },
        collections() {
            var colls = new Set();
            this.allitems.forEach(element => {
                if (element.collection != "" && element.collection != null)
                    colls.add(element.collection)
            });
            var cols = []
            colls.forEach(element => {
                    cols.push({
                        text: element,
                        value: element
                    })
                }

            )
            cols.sort((a, b) => {
                return a.text > b.text
            })
            return [{
                text: 'Show All',
                value: null
            }].concat(cols);
        }

    },
    created() {
        //
    },
};
</script>

<style >
</style>
